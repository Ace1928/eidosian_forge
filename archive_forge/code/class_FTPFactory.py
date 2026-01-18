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
class FTPFactory(policies.LimitTotalConnectionsFactory):
    """
    A factory for producing ftp protocol instances

    @ivar timeOut: the protocol interpreter's idle timeout time in seconds,
        default is 600 seconds.

    @ivar passivePortRange: value forwarded to C{protocol.passivePortRange}.
    @type passivePortRange: C{iterator}
    """
    protocol = FTP
    overflowProtocol = FTPOverflowProtocol
    allowAnonymous = True
    userAnonymous = 'anonymous'
    timeOut = 600
    welcomeMessage = f'Twisted {copyright.version} FTP Server'
    passivePortRange = range(0, 1)

    def __init__(self, portal=None, userAnonymous='anonymous'):
        self.portal = portal
        self.userAnonymous = userAnonymous
        self.instances = []

    def buildProtocol(self, addr):
        p = policies.LimitTotalConnectionsFactory.buildProtocol(self, addr)
        if p is not None:
            p.wrappedProtocol.portal = self.portal
            p.wrappedProtocol.timeOut = self.timeOut
            p.wrappedProtocol.passivePortRange = self.passivePortRange
        return p

    def stopFactory(self):
        [p.setTimeout(None) for p in self.instances if p.timeOut is not None]
        policies.LimitTotalConnectionsFactory.stopFactory(self)