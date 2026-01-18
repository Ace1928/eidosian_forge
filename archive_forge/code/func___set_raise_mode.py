from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
def __set_raise_mode(self, mode):
    assert mode in (RaiseMode.NONE, RaiseMode.ERRORS, RaiseMode.ALL)
    self._raise_mode = mode