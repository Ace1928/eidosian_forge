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
class IFinishableConsumer(interfaces.IConsumer):
    """
    A Consumer for producers that finish.

    @since: 11.0
    """

    def finish():
        """
        The producer has finished producing.
        """