import asyncio
import atexit
import time
import typing as t
from queue import Empty
from threading import Event, Thread
import zmq.asyncio
from jupyter_core.utils import ensure_async
from ._version import protocol_version_info
from .channelsabc import HBChannelABC
from .session import Session
class HBChannel(Thread):
    """The heartbeat channel which monitors the kernel heartbeat.

    Note that the heartbeat channel is paused by default. As long as you start
    this channel, the kernel manager will ensure that it is paused and un-paused
    as appropriate.
    """
    session = None
    socket = None
    address = None
    _exiting = False
    time_to_dead: float = 1.0
    _running = None
    _pause = None
    _beating = None

    def __init__(self, context: t.Optional[zmq.Context]=None, session: t.Optional[Session]=None, address: t.Union[t.Tuple[str, int], str]='') -> None:
        """Create the heartbeat monitor thread.

        Parameters
        ----------
        context : :class:`zmq.Context`
            The ZMQ context to use.
        session : :class:`session.Session`
            The session to use.
        address : zmq url
            Standard (ip, port) tuple that the kernel is listening on.
        """
        super().__init__()
        self.daemon = True
        self.context = context
        self.session = session
        if isinstance(address, tuple):
            if address[1] == 0:
                message = 'The port number for a channel cannot be 0.'
                raise InvalidPortNumber(message)
            address_str = 'tcp://%s:%i' % address
        else:
            address_str = address
        self.address = address_str
        self._running = False
        self._exit = Event()
        self._pause = False
        self.poller = zmq.Poller()

    @staticmethod
    @atexit.register
    def _notice_exit() -> None:
        if HBChannel is not None:
            HBChannel._exiting = True

    def _create_socket(self) -> None:
        if self.socket is not None:
            self.poller.unregister(self.socket)
            self.socket.close()
        assert self.context is not None
        self.socket = self.context.socket(zmq.REQ)
        self.socket.linger = 1000
        assert self.address is not None
        self.socket.connect(self.address)
        self.poller.register(self.socket, zmq.POLLIN)

    async def _async_run(self) -> None:
        """The thread's main activity.  Call start() instead."""
        self._create_socket()
        self._running = True
        self._beating = True
        assert self.socket is not None
        while self._running:
            if self._pause:
                self._exit.wait(self.time_to_dead)
                continue
            since_last_heartbeat = 0.0
            await ensure_async(self.socket.send(b'ping'))
            request_time = time.time()
            self._exit.wait(self.time_to_dead)
            self._beating = bool(self.poller.poll(0))
            if self._beating:
                await ensure_async(self.socket.recv())
                continue
            elif self._running:
                since_last_heartbeat = time.time() - request_time
                self.call_handlers(since_last_heartbeat)
                self._create_socket()
                continue

    def run(self) -> None:
        """Run the heartbeat thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_run())
        loop.close()

    def pause(self) -> None:
        """Pause the heartbeat."""
        self._pause = True

    def unpause(self) -> None:
        """Unpause the heartbeat."""
        self._pause = False

    def is_beating(self) -> bool:
        """Is the heartbeat running and responsive (and not paused)."""
        if self.is_alive() and (not self._pause) and self._beating:
            return True
        else:
            return False

    def stop(self) -> None:
        """Stop the channel's event loop and join its thread."""
        self._running = False
        self._exit.set()
        self.join()
        self.close()

    def close(self) -> None:
        """Close the heartbeat thread."""
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
            self.socket = None

    def call_handlers(self, since_last_heartbeat: float) -> None:
        """This method is called in the ioloop thread when a message arrives.

        Subclasses should override this method to handle incoming messages.
        It is important to remember that this method is called in the thread
        so that some logic must be done to ensure that the application level
        handlers are called in the application thread.
        """
        pass