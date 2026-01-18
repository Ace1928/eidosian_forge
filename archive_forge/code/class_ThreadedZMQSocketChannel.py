import asyncio
import atexit
import time
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional
import zmq
from tornado.ioloop import IOLoop
from traitlets import Instance, Type
from traitlets.log import get_logger
from zmq.eventloop import zmqstream
from .channels import HBChannel
from .client import KernelClient
from .session import Session
class ThreadedZMQSocketChannel:
    """A ZMQ socket invoking a callback in the ioloop"""
    session = None
    socket = None
    ioloop = None
    stream = None
    _inspect = None

    def __init__(self, socket: Optional[zmq.Socket], session: Optional[Session], loop: Optional[IOLoop]) -> None:
        """Create a channel.

        Parameters
        ----------
        socket : :class:`zmq.Socket`
            The ZMQ socket to use.
        session : :class:`session.Session`
            The session to use.
        loop
            A tornado ioloop to connect the socket to using a ZMQStream
        """
        super().__init__()
        self.socket = socket
        self.session = session
        self.ioloop = loop
        f: Future = Future()

        def setup_stream() -> None:
            try:
                assert self.socket is not None
                self.stream = zmqstream.ZMQStream(self.socket, self.ioloop)
                self.stream.on_recv(self._handle_recv)
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(None)
        assert self.ioloop is not None
        self.ioloop.add_callback(setup_stream)
        f.result(timeout=10)
    _is_alive = False

    def is_alive(self) -> bool:
        """Whether the channel is alive."""
        return self._is_alive

    def start(self) -> None:
        """Start the channel."""
        self._is_alive = True

    def stop(self) -> None:
        """Stop the channel."""
        self._is_alive = False

    def close(self) -> None:
        """Close the channel."""
        if self.stream is not None and self.ioloop is not None:
            f: Future = Future()

            def close_stream() -> None:
                try:
                    if self.stream is not None:
                        self.stream.close(linger=0)
                        self.stream = None
                except Exception as e:
                    f.set_exception(e)
                else:
                    f.set_result(None)
            self.ioloop.add_callback(close_stream)
            try:
                f.result(timeout=5)
            except Exception as e:
                log = get_logger()
                msg = f'Error closing stream {self.stream}: {e}'
                log.warning(msg, RuntimeWarning, stacklevel=2)
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self.socket = None

    def send(self, msg: Dict[str, Any]) -> None:
        """Queue a message to be sent from the IOLoop's thread.

        Parameters
        ----------
        msg : message to send

        This is threadsafe, as it uses IOLoop.add_callback to give the loop's
        thread control of the action.
        """

        def thread_send() -> None:
            assert self.session is not None
            self.session.send(self.stream, msg)
        assert self.ioloop is not None
        self.ioloop.add_callback(thread_send)

    def _handle_recv(self, msg_list: List) -> None:
        """Callback for stream.on_recv.

        Unpacks message, and calls handlers with it.
        """
        assert self.ioloop is not None
        assert self.session is not None
        ident, smsg = self.session.feed_identities(msg_list)
        msg = self.session.deserialize(smsg)
        if self._inspect:
            self._inspect(msg)
        self.call_handlers(msg)

    def call_handlers(self, msg: Dict[str, Any]) -> None:
        """This method is called in the ioloop thread when a message arrives.

        Subclasses should override this method to handle incoming messages.
        It is important to remember that this method is called in the thread
        so that some logic must be done to ensure that the application level
        handlers are called in the application thread.
        """
        pass

    def process_events(self) -> None:
        """Subclasses should override this with a method
        processing any pending GUI events.
        """
        pass

    def flush(self, timeout: float=1.0) -> None:
        """Immediately processes all pending messages on this channel.

        This is only used for the IOPub channel.

        Callers should use this method to ensure that :meth:`call_handlers`
        has been called for all messages that have been received on the
        0MQ SUB socket of this channel.

        This method is thread safe.

        Parameters
        ----------
        timeout : float, optional
            The maximum amount of time to spend flushing, in seconds. The
            default is one second.
        """
        stop_time = time.monotonic() + timeout
        assert self.ioloop is not None
        if self.stream is None or self.stream.closed():
            _msg = 'Attempt to flush closed stream'
            raise OSError(_msg)

        def flush(f: Any) -> None:
            try:
                self._flush()
            except Exception as e:
                f.set_exception(e)
            else:
                f.set_result(None)
        for _ in range(2):
            f: Future = Future()
            self.ioloop.add_callback(partial(flush, f))
            timeout = max(stop_time - time.monotonic(), 0)
            try:
                f.result(max(stop_time - time.monotonic(), 0))
            except TimeoutError:
                return

    def _flush(self) -> None:
        """Callback for :method:`self.flush`."""
        assert self.stream is not None
        self.stream.flush()
        self._flushed = True