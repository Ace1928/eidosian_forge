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
def __cbManualSearch(self, result, tag, mbox, query, uid, searchResults=None):
    """
        Apply the search filter to a set of messages. Send the response to the
        client.

        @type result: L{list} of L{tuple} of (L{int}, provider of
            L{imap4.IMessage})
        @param result: A list two tuples of messages with their sequence ids,
            sorted by the ids in descending order.

        @type tag: L{str}
        @param tag: A command tag.

        @type mbox: Provider of L{imap4.IMailbox}
        @param mbox: The searched mailbox.

        @type query: L{list}
        @param query: A list representing the parsed form of the search query.

        @param uid: A flag indicating whether the search is over message
            sequence numbers or UIDs.

        @type searchResults: L{list}
        @param searchResults: The search results so far or L{None} if no
            results yet.
        """
    if searchResults is None:
        searchResults = []
    i = 0
    lastSequenceId = result and result[-1][0]
    lastMessageId = result and result[-1][1].getUID()
    for i, (msgId, msg) in list(zip(range(5), result)):
        if self._searchFilter(copy.deepcopy(query), msgId, msg, lastSequenceId, lastMessageId):
            searchResults.append(b'%d' % (msg.getUID() if uid else msgId,))
    if i == 4:
        from twisted.internet import reactor
        reactor.callLater(0, self.__cbManualSearch, list(result[5:]), tag, mbox, query, uid, searchResults)
    else:
        if searchResults:
            self.sendUntaggedResponse(b'SEARCH ' + b' '.join(searchResults))
        self.sendPositiveResponse(tag, b'SEARCH completed')