import io
import os
import socket
import stat
from hashlib import md5
from typing import IO
from zope.interface import implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, interfaces, reactor
from twisted.mail import mail, pop3, smtp
from twisted.persisted import dirdbm
from twisted.protocols import basic
from twisted.python import failure, log
@implementer(pop3.IMailbox)
class StringListMailbox:
    """
    An in-memory mailbox.

    @ivar  msgs: See L{__init__}.

    @type _delete: L{set} of L{int}
    @ivar _delete: The indices of messages which have been marked for deletion.
    """

    def __init__(self, msgs):
        """
        @type msgs: L{list} of L{bytes}
        @param msgs: The contents of each message in the mailbox.
        """
        self.msgs = msgs
        self._delete = set()

    def listMessages(self, i=None):
        """
        Retrieve the size of a message, or, if none is specified, the size of
        each message in the mailbox.

        @type i: L{int} or L{None}
        @param i: The 0-based index of a message.

        @rtype: L{int} or L{list} of L{int}
        @return: The number of octets in the specified message, or, if an index
            is not specified, a list of the number of octets in each message in
            the mailbox.  Any value which corresponds to a deleted message is
            set to 0.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        if i is None:
            return [self.listMessages(msg) for msg in range(len(self.msgs))]
        if i in self._delete:
            return 0
        return len(self.msgs[i])

    def getMessage(self, i: int) -> IO[bytes]:
        """
        Return an in-memory file-like object with the contents of a message.

        @param i: The 0-based index of a message.

        @return: An in-memory file-like object containing the message.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        return io.BytesIO(self.msgs[i])

    def getUidl(self, i):
        """
        Get a unique identifier for a message.

        @type i: L{int}
        @param i: The 0-based index of a message.

        @rtype: L{bytes}
        @return: A hash of the contents of the message at the given index.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        return md5(self.msgs[i]).hexdigest()

    def deleteMessage(self, i):
        """
        Mark a message for deletion.

        @type i: L{int}
        @param i: The 0-based index of a message to delete.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        self._delete.add(i)

    def undeleteMessages(self):
        """
        Undelete any messages which have been marked for deletion.
        """
        self._delete = set()

    def sync(self):
        """
        Discard the contents of any messages marked for deletion.
        """
        for index in self._delete:
            self.msgs[index] = ''
        self._delete = set()