import asyncio
from threading import Event, Thread
from typing import Any, List, Optional
import zmq
import zmq.asyncio
from .base import Authenticator
class AuthenticationThread(Thread):
    """A Thread for running a zmq Authenticator

    This is run in the background by ThreadAuthenticator
    """
    pipe: zmq.Socket
    loop: asyncio.AbstractEventLoop
    authenticator: Authenticator
    poller: Optional[zmq.asyncio.Poller] = None

    def __init__(self, authenticator: Authenticator, pipe: zmq.Socket) -> None:
        super().__init__(daemon=True)
        self.authenticator = authenticator
        self.log = authenticator.log
        self.pipe = pipe
        self.started = Event()

    def run(self) -> None:
        """Start the Authentication Agent thread task"""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._run())
        finally:
            if self.pipe:
                self.pipe.close()
                self.pipe = None
            loop.close()

    async def _run(self):
        self.poller = zmq.asyncio.Poller()
        self.poller.register(self.pipe, zmq.POLLIN)
        self.poller.register(self.authenticator.zap_socket, zmq.POLLIN)
        self.started.set()
        while True:
            events = dict(await self.poller.poll())
            if self.pipe in events:
                msg = self.pipe.recv_multipart()
                if self._handle_pipe_message(msg):
                    return
            if self.authenticator.zap_socket in events:
                msg = self.authenticator.zap_socket.recv_multipart()
                await self.authenticator.handle_zap_message(msg)

    def _handle_pipe_message(self, msg: List[bytes]) -> bool:
        command = msg[0]
        self.log.debug('auth received API command %r', command)
        if command == b'TERMINATE':
            return True
        else:
            self.log.error('Invalid auth command from API: %r', command)
            self.pipe.send(b'ERROR')
        return False