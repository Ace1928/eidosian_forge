from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
class CheckersMixin:
    """
    L{unittest.TestCase} mixin for testing that some checkers accept
    and deny specified credentials.

    Subclasses must provide
      - C{getCheckers} which returns a sequence of
        L{checkers.ICredentialChecker}
      - C{getGoodCredentials} which returns a list of 2-tuples of
        credential to check and avaterId to expect.
      - C{getBadCredentials} which returns a list of credentials
        which are expected to be unauthorized.
    """

    @defer.inlineCallbacks
    def test_positive(self):
        """
        The given credentials are accepted by all the checkers, and give
        the expected C{avatarID}s
        """
        for chk in self.getCheckers():
            for cred, avatarId in self.getGoodCredentials():
                r = (yield chk.requestAvatarId(cred))
                self.assertEqual(r, avatarId)

    @defer.inlineCallbacks
    def test_negative(self):
        """
        The given credentials are rejected by all the checkers.
        """
        for chk in self.getCheckers():
            for cred in self.getBadCredentials():
                d = chk.requestAvatarId(cred)
                yield self.assertFailure(d, error.UnauthorizedLogin)