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
def _singleSearchStep(self, query, msgId, msg, lastSequenceId, lastMessageId):
    """
        Pop one search term from the beginning of C{query} (possibly more than
        one element) and return whether it matches the given message.

        @param query: A list representing the parsed form of the search query.

        @param msgId: The sequence number of the message being checked.

        @param msg: The message being checked.

        @param lastSequenceId: The highest sequence number of any message in
            the mailbox being searched.

        @param lastMessageId: The highest UID of any message in the mailbox
            being searched.

        @return: Boolean indicating whether the query term matched the message.
        """
    q = query.pop(0)
    if isinstance(q, list):
        if not self._searchFilter(q, msgId, msg, lastSequenceId, lastMessageId):
            return False
    else:
        c = q.upper()
        if not c[:1].isalpha():
            messageSet = parseIdList(c, lastSequenceId)
            return msgId in messageSet
        else:
            f = getattr(self, 'search_' + nativeString(c), None)
            if f is None:
                raise IllegalQueryError('Invalid search command %s' % nativeString(c))
            if c in self._requiresLastMessageInfo:
                result = f(query, msgId, msg, (lastSequenceId, lastMessageId))
            else:
                result = f(query, msgId, msg)
            if not result:
                return False
    return True