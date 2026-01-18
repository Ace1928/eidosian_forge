import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def __doCommand(self, tag, handler, args, parseargs, line, uid):
    for i, arg in enumerate(parseargs):
        if callable(arg):
            parseargs = parseargs[i + 1:]
            maybeDeferred(arg, self, line).addCallback(self.__cbDispatch, tag, handler, args, parseargs, uid).addErrback(self.__ebDispatch, tag)
            return
        else:
            args.append(arg)
    if line:
        raise IllegalClientResponse('Too many arguments for command: ' + repr(line))
    if uid is not None:
        handler(*args, uid=uid)
    else:
        handler(*args)