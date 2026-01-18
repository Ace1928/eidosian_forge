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
def __cbSelect(self, result, rw):
    """
        Handle lines received in response to a SELECT or EXAMINE command.

        See RFC 3501, section 6.3.1.
        """
    lines, tagline = result
    datum = {'READ-WRITE': rw}
    lines.append(parseNestedParens(tagline))
    for split in lines:
        if len(split) > 0 and split[0].upper() == b'OK':
            content = split[1]
            if isinstance(content, list):
                key = content[0]
            else:
                key = content
            key = key.upper()
            if key == b'READ-ONLY':
                datum['READ-WRITE'] = False
            elif key == b'READ-WRITE':
                datum['READ-WRITE'] = True
            elif key == b'UIDVALIDITY':
                datum['UIDVALIDITY'] = self._intOrRaise(content[1], split)
            elif key == b'UNSEEN':
                datum['UNSEEN'] = self._intOrRaise(content[1], split)
            elif key == b'UIDNEXT':
                datum['UIDNEXT'] = self._intOrRaise(content[1], split)
            elif key == b'PERMANENTFLAGS':
                datum['PERMANENTFLAGS'] = tuple((nativeString(flag) for flag in content[1]))
            else:
                log.err(f'Unhandled SELECT response (2): {split}')
        elif len(split) == 2:
            if split[0].upper() == b'FLAGS':
                datum['FLAGS'] = tuple((nativeString(flag) for flag in split[1]))
            elif isinstance(split[1], bytes):
                if split[1].upper() == b'EXISTS':
                    datum['EXISTS'] = self._intOrRaise(split[0], split)
                elif split[1].upper() == b'RECENT':
                    datum['RECENT'] = self._intOrRaise(split[0], split)
                else:
                    log.err(f'Unhandled SELECT response (0): {split}')
            else:
                log.err(f'Unhandled SELECT response (1): {split}')
        else:
            log.err(f'Unhandled SELECT response (4): {split}')
    return datum