import errno
import sys
from asyncio import AbstractEventLoop, get_event_loop
from typing import Dict, Optional, Type
from zope.interface import implementer
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import (
from twisted.logger import Logger
from twisted.python.log import callWithLogger
def _onTimer(self):
    self._scheduledAt = None
    self.runUntilCurrent()
    self._reschedule()