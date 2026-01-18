from twisted.trial import unittest
from twisted.words.protocols.jabber.xmpp_stringprep import (
class XMPPStringPrepTests(unittest.TestCase):
    """
    The nodeprep stringprep profile is similar to the resourceprep profile,
    but does an extra mapping of characters (table B.2) and disallows
    more characters (table C.1.1 and eight extra punctuation characters).
    Due to this similarity, the resourceprep tests are more extensive, and
    the nodeprep tests only address the mappings additional restrictions.

    The nameprep profile is nearly identical to the nameprep implementation in
    L{encodings.idna}, but that implementation assumes the C{UseSTD4ASCIIRules}
    flag to be false. This implementation assumes it to be true, and restricts
    the allowed set of characters.  The tests here only check for the
    differences.
    """

    def testResourcePrep(self) -> None:
        self.assertEqual(resourceprep.prepare('resource'), 'resource')
        self.assertNotEqual(resourceprep.prepare('Resource'), 'resource')
        self.assertEqual(resourceprep.prepare(' '), ' ')
        self.assertEqual(resourceprep.prepare('Henry Ⅳ'), 'Henry IV')
        self.assertEqual(resourceprep.prepare('foo\xad͏᠆᠋bar\u200b\u2060baz︀︈️\ufeff'), 'foobarbaz')
        self.assertEqual(resourceprep.prepare('\xa0'), ' ')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\u1680')
        self.assertEqual(resourceprep.prepare('\u2000'), ' ')
        self.assertEqual(resourceprep.prepare('\u200b'), '')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\x10\x7f')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\x85')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\u180e')
        self.assertEqual(resourceprep.prepare('\ufeff'), '')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\uf123')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U000f1234')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U0010f234')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U0008fffe')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U0010ffff')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\udf42')
        self.assertRaises(UnicodeError, resourceprep.prepare, '�')
        self.assertRaises(UnicodeError, resourceprep.prepare, '⿵')
        self.assertEqual(resourceprep.prepare('́'), '́')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\u200e')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\u202a')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U000e0001')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U000e0042')
        self.assertRaises(UnicodeError, resourceprep.prepare, 'foo־bar')
        self.assertRaises(UnicodeError, resourceprep.prepare, 'fooﵐbar')
        self.assertRaises(UnicodeError, resourceprep.prepare, 'ا1')
        self.assertEqual(resourceprep.prepare('ا1ب'), 'ا1ب')
        self.assertRaises(UnicodeError, resourceprep.prepare, '\U000e0002')

    def testNodePrep(self) -> None:
        self.assertEqual(nodeprep.prepare('user'), 'user')
        self.assertEqual(nodeprep.prepare('User'), 'user')
        self.assertRaises(UnicodeError, nodeprep.prepare, 'us&er')

    def test_nodeprepUnassignedInUnicode32(self) -> None:
        """
        Make sure unassigned code points from Unicode 3.2 are rejected.
        """
        self.assertRaises(UnicodeError, nodeprep.prepare, 'ᴹ')

    def testNamePrep(self) -> None:
        self.assertEqual(nameprep.prepare('example.com'), 'example.com')
        self.assertEqual(nameprep.prepare('Example.com'), 'example.com')
        self.assertRaises(UnicodeError, nameprep.prepare, 'ex@mple.com')
        self.assertRaises(UnicodeError, nameprep.prepare, '-example.com')
        self.assertRaises(UnicodeError, nameprep.prepare, 'example-.com')
        self.assertEqual(nameprep.prepare('straße.example.com'), 'strasse.example.com')

    def test_nameprepTrailingDot(self) -> None:
        """
        A trailing dot in domain names is preserved.
        """
        self.assertEqual(nameprep.prepare('example.com.'), 'example.com.')