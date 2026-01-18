from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
class InternJIDTests(unittest.TestCase):

    def test_identity(self) -> None:
        """
        Test that two interned JIDs yield the same object.
        """
        j1 = jid.internJID('user@host')
        j2 = jid.internJID('user@host')
        self.assertIdentical(j1, j2)