import msvcrt
import os
from zope.interface import implementer
import win32api
from twisted.internet import _pollingfile, main
from twisted.internet.interfaces import (
from twisted.python.failure import Failure

        Start talking to standard IO with the given protocol.

        Also, put it stdin/stdout/stderr into binary mode.
        