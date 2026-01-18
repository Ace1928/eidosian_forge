import time
from twisted.cred import checkers, credentials, portal
from twisted.internet import address, defer, reactor
from twisted.internet.defer import Deferred, DeferredList, maybeDeferred, succeed
from twisted.spread import pb
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words import ewords, service
from twisted.words.protocols import irc
def _loggedInUser(self, name):
    user = self.successResultOf(self.realm.lookupUser(name))
    agg = TestCaseUserAgg(user, self.realm, self.factory)
    self._login(agg, name)
    return agg