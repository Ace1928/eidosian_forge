import errno
import fnmatch
import os
import re
import stat
import time
from zope.interface import Interface, implementer
from twisted import copyright
from twisted.cred import checkers, credentials, error as cred_error, portal
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.protocols import basic, policies
from twisted.python import failure, filepath, log
class IWriteFile(Interface):
    """
    A file into which bytes may be written.
    """

    def receive():
        """
        Create a consumer which will write to this file.  This method may
        only be invoked once on each provider.

        @rtype: C{Deferred} of C{IConsumer}
        """

    def close():
        """
        Perform any post-write work that needs to be done. This method may
        only be invoked once on each provider, and will always be invoked
        after receive().

        @rtype: C{Deferred} of anything: the value is ignored. The FTP client
        will not see their upload request complete until this Deferred has
        been fired.
        """