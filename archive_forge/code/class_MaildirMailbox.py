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
class MaildirMailbox(pop3.Mailbox):
    """
    A maildir-backed mailbox.

    @ivar path: See L{__init__}.

    @type list: L{list} of L{int} or 2-L{tuple} of (0) file-like object,
        (1) L{bytes}
    @ivar list: Information about the messages in the mailbox. For undeleted
        messages, the file containing the message and the
        full path name of the file are stored.  Deleted messages are indicated
        by 0.

    @type deleted: L{dict} mapping 2-L{tuple} of (0) file-like object,
        (1) L{bytes} to L{bytes}
    @type deleted: A mapping of the information about a file before it was
        deleted to the full path name of the deleted file in the I{.Trash/}
        subfolder.
    """
    AppendFactory = _MaildirMailboxAppendMessageTask

    def __init__(self, path):
        """
        @type path: L{bytes}
        @param path: The directory name for a maildir mailbox.
        """
        self.path = path
        self.list = []
        self.deleted = {}
        initializeMaildir(path)
        for name in ('cur', 'new'):
            for file in os.listdir(os.path.join(path, name)):
                self.list.append((file, os.path.join(path, name, file)))
        self.list.sort()
        self.list = [e[1] for e in self.list]

    def listMessages(self, i=None):
        """
        Retrieve the size of a message, or, if none is specified, the size of
        each message in the mailbox.

        @type i: L{int} or L{None}
        @param i: The 0-based index of a message.

        @rtype: L{int} or L{list} of L{int}
        @return: The number of octets in the specified message, or, if an index
            is not specified, a list of the number of octets for all messages
            in the mailbox.  Any value which corresponds to a deleted message
            is set to 0.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        if i is None:
            ret = []
            for mess in self.list:
                if mess:
                    ret.append(os.stat(mess)[stat.ST_SIZE])
                else:
                    ret.append(0)
            return ret
        return self.list[i] and os.stat(self.list[i])[stat.ST_SIZE] or 0

    def getMessage(self, i):
        """
        Retrieve a file-like object with the contents of a message.

        @type i: L{int}
        @param i: The 0-based index of a message.

        @rtype: file-like object
        @return: A file containing the message.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        return open(self.list[i])

    def getUidl(self, i):
        """
        Get a unique identifier for a message.

        @type i: L{int}
        @param i: The 0-based index of a message.

        @rtype: L{bytes}
        @return: A string of printable characters uniquely identifying the
            message for all time.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        base = os.path.basename(self.list[i])
        return md5(base).hexdigest()

    def deleteMessage(self, i):
        """
        Mark a message for deletion.

        Move the message to the I{.Trash/} subfolder so it can be undeleted
        by an administrator.

        @type i: L{int}
        @param i: The 0-based index of a message.

        @raise IndexError: When the index does not correspond to a message in
            the mailbox.
        """
        trashFile = os.path.join(self.path, '.Trash', 'cur', os.path.basename(self.list[i]))
        os.rename(self.list[i], trashFile)
        self.deleted[self.list[i]] = trashFile
        self.list[i] = 0

    def undeleteMessages(self):
        """
        Undelete all messages marked for deletion.

        Move each message marked for deletion from the I{.Trash/} subfolder back
        to its original position.
        """
        for real, trash in self.deleted.items():
            try:
                os.rename(trash, real)
            except OSError as e:
                err, estr = e.args
                import errno
                if err != errno.ENOENT:
                    raise
            else:
                try:
                    self.list[self.list.index(0)] = real
                except ValueError:
                    self.list.append(real)
        self.deleted.clear()

    def appendMessage(self, txt):
        """
        Add a message to the mailbox.

        @type txt: L{bytes} or file-like object
        @param txt: A message to add.

        @rtype: L{Deferred <defer.Deferred>}
        @return: A deferred which fires when the message has been added to
            the mailbox.
        """
        task = self.AppendFactory(self, txt)
        result = task.defer
        task.startUp()
        return result