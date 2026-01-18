from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def _wakerFactory(self) -> _WakerPlus:
    return _WakerPlus(self)