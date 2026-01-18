import asyncio
from threading import Event, Thread
from typing import Any, List, Optional
import zmq
import zmq.asyncio
from .base import Authenticator
def _handle_pipe_message(self, msg: List[bytes]) -> bool:
    command = msg[0]
    self.log.debug('auth received API command %r', command)
    if command == b'TERMINATE':
        return True
    else:
        self.log.error('Invalid auth command from API: %r', command)
        self.pipe.send(b'ERROR')
    return False