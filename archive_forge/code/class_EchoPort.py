import random
import threading
import time
from .messages import Message
from .parser import Parser
class EchoPort(BaseIOPort):

    def _send(self, message):
        self._messages.append(message)
    __iter__ = BaseIOPort.iter_pending