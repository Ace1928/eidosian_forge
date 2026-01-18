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
def _cbFetch(self, result, requestedParts, structured):
    lines, last = result
    info = {}
    for parts in lines:
        if len(parts) == 3 and parts[1] == b'FETCH':
            id = self._intOrRaise(parts[0], parts)
            if id not in info:
                info[id] = [parts[2]]
            else:
                info[id][0].extend(parts[2])
    results = {}
    decodedInfo = {}
    for messageId, values in info.items():
        structuredMap, unstructuredList = self._parseFetchPairs(values[0])
        decodedInfo.setdefault(messageId, [[]])[0].extend(unstructuredList)
        results.setdefault(messageId, {}).update(structuredMap)
    info = decodedInfo
    flagChanges = {}
    for messageId in list(results.keys()):
        values = results[messageId]
        for part in list(values.keys()):
            if part not in requestedParts and part == 'FLAGS':
                flagChanges[messageId] = values['FLAGS']
                for i in range(len(info[messageId][0])):
                    if info[messageId][0][i] == 'FLAGS':
                        del info[messageId][0][i:i + 2]
                        break
                del values['FLAGS']
                if not values:
                    del results[messageId]
    if flagChanges:
        self.flagsChanged(flagChanges)
    if structured:
        return results
    else:
        return info