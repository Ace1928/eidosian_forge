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
def _searchFilter(self, query, id, msg, lastSequenceId, lastMessageId):
    """
        Pop search terms from the beginning of C{query} until there are none
        left and apply them to the given message.

        @param query: A list representing the parsed form of the search query.

        @param id: The sequence number of the message being checked.

        @param msg: The message being checked.

        @type lastSequenceId: L{int}
        @param lastSequenceId: The highest sequence number of any message in
            the mailbox being searched.

        @type lastMessageId: L{int}
        @param lastMessageId: The highest UID of any message in the mailbox
            being searched.

        @return: Boolean indicating whether all of the query terms match the
            message.
        """
    while query:
        if not self._singleSearchStep(query, id, msg, lastSequenceId, lastMessageId):
            return False
    return True