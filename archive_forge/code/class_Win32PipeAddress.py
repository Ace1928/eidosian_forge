import msvcrt
import os
from zope.interface import implementer
import win32api
from twisted.internet import _pollingfile, main
from twisted.internet.interfaces import (
from twisted.python.failure import Failure
@implementer(IAddress)
class Win32PipeAddress:
    pass