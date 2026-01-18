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
def __cbStore(self, result, tag, mbox, uid, silent):
    if result and (not silent):
        for k, v in result.items():
            if uid:
                uidstr = b' UID %d' % (mbox.getUID(k),)
            else:
                uidstr = b''
            flags = [networkString(flag) for flag in v]
            self.sendUntaggedResponse(b'%d FETCH (FLAGS (%b)%b)' % (k, b' '.join(flags), uidstr))
    self.sendPositiveResponse(tag, b'STORE completed')