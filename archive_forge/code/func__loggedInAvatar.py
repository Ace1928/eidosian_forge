import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def _loggedInAvatar(self, name, password, mind):
    nameBytes = name
    if isinstance(name, str):
        nameBytes = name.encode('ascii')
    creds = credentials.UsernamePassword(nameBytes, password)
    self.checker.addUser(nameBytes, password)
    d = self.realm.createUser(name)
    d.addCallback(lambda ign: self.clientFactory.login(creds, mind))
    return d