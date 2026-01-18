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
class FTPCommand:

    def __init__(self, text=None, public=0):
        self.text = text
        self.deferred = defer.Deferred()
        self.ready = 1
        self.public = public
        self.transferDeferred = None

    def fail(self, failure):
        if self.public:
            self.deferred.errback(failure)