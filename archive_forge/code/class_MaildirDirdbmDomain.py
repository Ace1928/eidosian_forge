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
@implementer(portal.IRealm)
class MaildirDirdbmDomain(AbstractMaildirDomain):
    """
    A maildir-backed domain where membership is checked with a
    L{DirDBM <dirdbm.DirDBM>} database.

    The directory structure of a MaildirDirdbmDomain is:

    /passwd <-- a DirDBM directory

    /USER/{cur, new, del} <-- each user has these three directories

    @ivar postmaster: See L{__init__}.

    @type dbm: L{DirDBM <dirdbm.DirDBM>}
    @ivar dbm: The authentication database for the domain.
    """
    portal = None
    _credcheckers = None

    def __init__(self, service, root, postmaster=0):
        """
        @type service: L{MailService}
        @param service: An email service.

        @type root: L{bytes}
        @param root: The maildir root directory.

        @type postmaster: L{bool}
        @param postmaster: A flag indicating whether non-existent addresses
            should be forwarded to the postmaster (C{True}) or
            bounced (C{False}).
        """
        root = os.fsencode(root)
        AbstractMaildirDomain.__init__(self, service, root)
        dbm = os.path.join(root, b'passwd')
        if not os.path.exists(dbm):
            os.makedirs(dbm)
        self.dbm = dirdbm.open(dbm)
        self.postmaster = postmaster

    def userDirectory(self, name):
        """
        Return the path to a user's mail directory.

        @type name: L{bytes}
        @param name: A username.

        @rtype: L{bytes} or L{None}
        @return: The path to the user's mail directory for a valid user. For
            an invalid user, the path to the postmaster's mailbox if bounces
            are redirected there. Otherwise, L{None}.
        """
        if name not in self.dbm:
            if not self.postmaster:
                return None
            name = 'postmaster'
        dir = os.path.join(self.root, name)
        if not os.path.exists(dir):
            initializeMaildir(dir)
        return dir

    def addUser(self, user, password):
        """
        Add a user to this domain by adding an entry in the authentication
        database and initializing the user's mail directory.

        @type user: L{bytes}
        @param user: A username.

        @type password: L{bytes}
        @param password: A password.
        """
        self.dbm[user] = password
        self.userDirectory(user)

    def getCredentialsCheckers(self):
        """
        Return credentials checkers for this domain.

        @rtype: L{list} of L{ICredentialsChecker
            <checkers.ICredentialsChecker>} provider
        @return: Credentials checkers for this domain.
        """
        if self._credcheckers is None:
            self._credcheckers = [DirdbmDatabase(self.dbm)]
        return self._credcheckers

    def requestAvatar(self, avatarId, mind, *interfaces):
        """
        Get the mailbox for an authenticated user.

        The mailbox for the authenticated user will be returned only if the
        given interfaces include L{IMailbox <pop3.IMailbox>}.  Requests for
        anonymous access will be met with a mailbox containing a message
        indicating that an internal error has occurred.

        @type avatarId: L{bytes} or C{twisted.cred.checkers.ANONYMOUS}
        @param avatarId: A string which identifies a user or an object which
            signals a request for anonymous access.

        @type mind: L{None}
        @param mind: Unused.

        @type interfaces: n-L{tuple} of C{zope.interface.Interface}
        @param interfaces: A group of interfaces, one of which the avatar
            must support.

        @rtype: 3-L{tuple} of (0) L{IMailbox <pop3.IMailbox>},
            (1) L{IMailbox <pop3.IMailbox>} provider, (2) no-argument
            callable
        @return: A tuple of the supported interface, a mailbox, and a
            logout function.

        @raise NotImplementedError: When the given interfaces do not include
            L{IMailbox <pop3.IMailbox>}.
        """
        if pop3.IMailbox not in interfaces:
            raise NotImplementedError('No interface')
        if avatarId == checkers.ANONYMOUS:
            mbox = StringListMailbox([INTERNAL_ERROR])
        else:
            mbox = MaildirMailbox(os.path.join(self.root, avatarId))
        return (pop3.IMailbox, mbox, lambda: None)