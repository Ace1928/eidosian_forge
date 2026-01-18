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
class IReadFile(Interface):
    """
    A file out of which bytes may be read.
    """

    def send(consumer):
        """
        Produce the contents of the given path to the given consumer.  This
        method may only be invoked once on each provider.

        @type consumer: C{IConsumer}

        @return: A Deferred which fires when the file has been
        consumed completely.
        """