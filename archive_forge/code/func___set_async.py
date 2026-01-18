from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
def __set_async(self, async_mode=True):
    self._async = async_mode
    if async_mode and (not self._session.can_pipeline):
        raise UserWarning('Asynchronous mode not supported for this device/session')