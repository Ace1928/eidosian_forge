import collections
import socket
import sys
import warnings
import weakref
from . import coroutines
from . import events
from . import exceptions
from . import format_helpers
from . import protocols
from .log import logger
from .tasks import sleep
class StreamReaderProtocol(FlowControlMixin, protocols.Protocol):
    """Helper class to adapt between Protocol and StreamReader.

    (This is a helper class instead of making StreamReader itself a
    Protocol subclass, because the StreamReader has other potential
    uses, and to prevent the user of the StreamReader to accidentally
    call inappropriate methods of the protocol.)
    """
    _source_traceback = None

    def __init__(self, stream_reader, client_connected_cb=None, loop=None):
        super().__init__(loop=loop)
        if stream_reader is not None:
            self._stream_reader_wr = weakref.ref(stream_reader)
            self._source_traceback = stream_reader._source_traceback
        else:
            self._stream_reader_wr = None
        if client_connected_cb is not None:
            self._strong_reader = stream_reader
        self._reject_connection = False
        self._stream_writer = None
        self._task = None
        self._transport = None
        self._client_connected_cb = client_connected_cb
        self._over_ssl = False
        self._closed = self._loop.create_future()

    @property
    def _stream_reader(self):
        if self._stream_reader_wr is None:
            return None
        return self._stream_reader_wr()

    def _replace_writer(self, writer):
        loop = self._loop
        transport = writer.transport
        self._stream_writer = writer
        self._transport = transport
        self._over_ssl = transport.get_extra_info('sslcontext') is not None

    def connection_made(self, transport):
        if self._reject_connection:
            context = {'message': 'An open stream was garbage collected prior to establishing network connection; call "stream.close()" explicitly.'}
            if self._source_traceback:
                context['source_traceback'] = self._source_traceback
            self._loop.call_exception_handler(context)
            transport.abort()
            return
        self._transport = transport
        reader = self._stream_reader
        if reader is not None:
            reader.set_transport(transport)
        self._over_ssl = transport.get_extra_info('sslcontext') is not None
        if self._client_connected_cb is not None:
            self._stream_writer = StreamWriter(transport, self, reader, self._loop)
            res = self._client_connected_cb(reader, self._stream_writer)
            if coroutines.iscoroutine(res):

                def callback(task):
                    if task.cancelled():
                        transport.close()
                        return
                    exc = task.exception()
                    if exc is not None:
                        self._loop.call_exception_handler({'message': 'Unhandled exception in client_connected_cb', 'exception': exc, 'transport': transport})
                        transport.close()
                self._task = self._loop.create_task(res)
                self._task.add_done_callback(callback)
            self._strong_reader = None

    def connection_lost(self, exc):
        reader = self._stream_reader
        if reader is not None:
            if exc is None:
                reader.feed_eof()
            else:
                reader.set_exception(exc)
        if not self._closed.done():
            if exc is None:
                self._closed.set_result(None)
            else:
                self._closed.set_exception(exc)
        super().connection_lost(exc)
        self._stream_reader_wr = None
        self._stream_writer = None
        self._task = None
        self._transport = None

    def data_received(self, data):
        reader = self._stream_reader
        if reader is not None:
            reader.feed_data(data)

    def eof_received(self):
        reader = self._stream_reader
        if reader is not None:
            reader.feed_eof()
        if self._over_ssl:
            return False
        return True

    def _get_close_waiter(self, stream):
        return self._closed

    def __del__(self):
        try:
            closed = self._closed
        except AttributeError:
            pass
        else:
            if closed.done() and (not closed.cancelled()):
                closed.exception()