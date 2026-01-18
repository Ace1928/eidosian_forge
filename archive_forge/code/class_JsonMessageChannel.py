from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
class JsonMessageChannel(object):
    """Implements a JSON message channel on top of a raw JSON message stream, with
    support for DAP requests, responses, and events.

    The channel can be locked for exclusive use via the with-statement::

        with channel:
            channel.send_request(...)
            # No interleaving messages can be sent here from other threads.
            channel.send_event(...)
    """

    def __init__(self, stream, handlers=None, name=None):
        self.stream = stream
        self.handlers = handlers
        self.name = name if name is not None else stream.name
        self.started = False
        self._lock = threading.RLock()
        self._closed = False
        self._seq_iter = itertools.count(1)
        self._sent_requests = {}
        self._handler_queue = []
        self._handlers_enqueued = threading.Condition(self._lock)
        self._handler_thread = None
        self._parser_thread = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'{type(self).__name__}({self.name!r})'

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._lock.release()

    def close(self):
        """Closes the underlying stream.

        This does not immediately terminate any handlers that are already executing,
        but they will be unable to respond. No new request or event handlers will
        execute after this method is called, even for messages that have already been
        received. However, response handlers will continue to executed for any request
        that is still pending, as will any handlers registered via on_response().
        """
        with self:
            if not self._closed:
                self._closed = True
                self.stream.close()

    def start(self):
        """Starts a message loop which parses incoming messages and invokes handlers
        for them on a background thread, until the channel is closed.

        Incoming messages, including responses to requests, will not be processed at
        all until this is invoked.
        """
        assert not self.started
        self.started = True
        self._parser_thread = threading.Thread(target=self._parse_incoming_messages, name=f'{self} message parser')
        hide_thread_from_debugger(self._parser_thread)
        self._parser_thread.daemon = True
        self._parser_thread.start()

    def wait(self):
        """Waits for the message loop to terminate, and for all enqueued Response
        message handlers to finish executing.
        """
        parser_thread = self._parser_thread
        try:
            if parser_thread is not None:
                parser_thread.join()
        except AssertionError:
            log.debug('Handled error joining parser thread.')
        try:
            handler_thread = self._handler_thread
            if handler_thread is not None:
                handler_thread.join()
        except AssertionError:
            log.debug('Handled error joining handler thread.')
    _prettify_order = ('seq', 'type', 'request_seq', 'success', 'command', 'event', 'message', 'arguments', 'body', 'error')

    def _prettify(self, message_dict):
        """Reorders items in a MessageDict such that it is more readable."""
        for key in self._prettify_order:
            if key not in message_dict:
                continue
            value = message_dict[key]
            del message_dict[key]
            message_dict[key] = value

    @contextlib.contextmanager
    def _send_message(self, message):
        """Sends a new message to the other party.

        Generates a new sequence number for the message, and provides it to the
        caller before the message is sent, using the context manager protocol::

            with send_message(...) as seq:
                # The message hasn't been sent yet.
                ...
            # Now the message has been sent.

        Safe to call concurrently for the same channel from different threads.
        """
        assert 'seq' not in message
        with self:
            seq = next(self._seq_iter)
        message = MessageDict(None, message)
        message['seq'] = seq
        self._prettify(message)
        with self:
            yield seq
            self.stream.write_json(message)

    def send_request(self, command, arguments=None, on_before_send=None):
        """Sends a new request, and returns the OutgoingRequest object for it.

        If arguments is None or {}, "arguments" will be omitted in JSON.

        If on_before_send is not None, invokes on_before_send() with the request
        object as the sole argument, before the request actually gets sent.

        Does not wait for response - use OutgoingRequest.wait_for_response().

        Safe to call concurrently for the same channel from different threads.
        """
        d = {'type': 'request', 'command': command}
        if arguments is not None and arguments != {}:
            d['arguments'] = arguments
        with self._send_message(d) as seq:
            request = OutgoingRequest(self, seq, command, arguments)
            if on_before_send is not None:
                on_before_send(request)
            self._sent_requests[seq] = request
        return request

    def send_event(self, event, body=None):
        """Sends a new event.

        If body is None or {}, "body" will be omitted in JSON.

        Safe to call concurrently for the same channel from different threads.
        """
        d = {'type': 'event', 'event': event}
        if body is not None and body != {}:
            d['body'] = body
        with self._send_message(d):
            pass

    def request(self, *args, **kwargs):
        """Same as send_request(...).wait_for_response()"""
        return self.send_request(*args, **kwargs).wait_for_response()

    def propagate(self, message):
        """Sends a new message with the same type and payload.

        If it was a request, returns the new OutgoingRequest object for it.
        """
        assert message.is_request() or message.is_event()
        if message.is_request():
            return self.send_request(message.command, message.arguments)
        else:
            self.send_event(message.event, message.body)

    def delegate(self, message):
        """Like propagate(message).wait_for_response(), but will also propagate
        any resulting MessageHandlingError back.
        """
        try:
            result = self.propagate(message)
            if result.is_request():
                result = result.wait_for_response()
            return result
        except MessageHandlingError as exc:
            exc.propagate(message)

    def _parse_incoming_messages(self):
        log.debug('Starting message loop for channel {0}', self)
        try:
            while True:
                self._parse_incoming_message()
        except NoMoreMessages as exc:
            log.debug('Exiting message loop for channel {0}: {1}', self, exc)
            with self:
                err_message = str(exc)
                sent_requests = list(self._sent_requests.values())
                for request in sent_requests:
                    response_json = MessageDict(None, {'seq': -1, 'request_seq': request.seq, 'command': request.command, 'success': False, 'message': err_message})
                    Response._parse(self, response_json, body=exc)
                assert not len(self._sent_requests)
                self._enqueue_handlers(Disconnect(self), self._handle_disconnect)
                self.close()
    _message_parsers = {'event': Event._parse, 'request': Request._parse, 'response': Response._parse}

    def _parse_incoming_message(self):
        """Reads incoming messages, parses them, and puts handlers into the queue
        for _run_handlers() to invoke, until the channel is closed.
        """

        def object_hook(d):
            d = MessageDict(None, d)
            if 'seq' in d:
                self._prettify(d)
            d.associate_with = associate_with
            message_dicts.append(d)
            return d

        def associate_with(message):
            for d in message_dicts:
                d.message = message
                del d.associate_with
        message_dicts = []
        decoder = self.stream.json_decoder_factory(object_hook=object_hook)
        message_dict = self.stream.read_json(decoder)
        assert isinstance(message_dict, MessageDict)
        msg_type = message_dict('type', json.enum('event', 'request', 'response'))
        parser = self._message_parsers[msg_type]
        try:
            parser(self, message_dict)
        except InvalidMessageError as exc:
            log.error('Failed to parse message in channel {0}: {1} in:\n{2}', self, str(exc), json.repr(message_dict))
        except Exception as exc:
            if isinstance(exc, NoMoreMessages) and exc.stream is self.stream:
                raise
            log.swallow_exception('Fatal error in channel {0} while parsing:\n{1}', self, json.repr(message_dict))
            os._exit(1)

    def _enqueue_handlers(self, what, *handlers):
        """Enqueues handlers for _run_handlers() to run.

        `what` is the Message being handled, and is used for logging purposes.

        If the background thread with _run_handlers() isn't running yet, starts it.
        """
        with self:
            self._handler_queue.extend(((what, handler) for handler in handlers))
            self._handlers_enqueued.notify_all()
            if len(self._handler_queue) and self._handler_thread is None:
                self._handler_thread = threading.Thread(target=self._run_handlers, name=f'{self} message handler')
                hide_thread_from_debugger(self._handler_thread)
                self._handler_thread.start()

    def _run_handlers(self):
        """Runs enqueued handlers until the channel is closed, or until the handler
        queue is empty once the channel is closed.
        """
        while True:
            with self:
                closed = self._closed
            if closed:
                self._parser_thread.join()
            with self:
                if not closed and (not len(self._handler_queue)):
                    self._handlers_enqueued.wait()
                handlers = self._handler_queue[:]
                del self._handler_queue[:]
                if closed and (not len(handlers)):
                    self._handler_thread = None
                    return
            for what, handler in handlers:
                if closed and handler in (Event._handle, Request._handle):
                    continue
                with log.prefixed('/handling {0}/\n', what.describe()):
                    try:
                        handler()
                    except Exception:
                        self.close()
                        os._exit(1)

    def _get_handler_for(self, type, name):
        """Returns the handler for a message of a given type."""
        with self:
            handlers = self.handlers
        for handler_name in (name + '_' + type, type):
            try:
                return getattr(handlers, handler_name)
            except AttributeError:
                continue
        raise AttributeError('handler object {0} for channel {1} has no handler for {2} {3!r}'.format(util.srcnameof(handlers), self, type, name))

    def _handle_disconnect(self):
        handler = getattr(self.handlers, 'disconnect', lambda: None)
        try:
            handler()
        except Exception:
            log.reraise_exception("Handler {0}\ncouldn't handle disconnect from {1}:", util.srcnameof(handler), self)