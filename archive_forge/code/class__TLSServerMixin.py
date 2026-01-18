from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
class _TLSServerMixin:
    pass