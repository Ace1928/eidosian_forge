import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class NameToLabelsTests(unittest.SynchronousTestCase):
    """
    Tests for L{twisted.names.dns._nameToLabels}.
    """

    def test_empty(self):
        """
        L{dns._nameToLabels} returns a list containing a single
        empty label for an empty name.
        """
        self.assertEqual(dns._nameToLabels(b''), [b''])

    def test_onlyDot(self):
        """
        L{dns._nameToLabels} returns a list containing a single
        empty label for a name containing only a dot.
        """
        self.assertEqual(dns._nameToLabels(b'.'), [b''])

    def test_withoutTrailingDot(self):
        """
        L{dns._nameToLabels} returns a list ending with an empty
        label for a name without a trailing dot.
        """
        self.assertEqual(dns._nameToLabels(b'com'), [b'com', b''])

    def test_withTrailingDot(self):
        """
        L{dns._nameToLabels} returns a list ending with an empty
        label for a name with a trailing dot.
        """
        self.assertEqual(dns._nameToLabels(b'com.'), [b'com', b''])

    def test_subdomain(self):
        """
        L{dns._nameToLabels} returns a list containing entries
        for all labels in a subdomain name.
        """
        self.assertEqual(dns._nameToLabels(b'foo.bar.baz.example.com.'), [b'foo', b'bar', b'baz', b'example', b'com', b''])

    def test_casePreservation(self):
        """
        L{dns._nameToLabels} preserves the case of ascii
        characters in labels.
        """
        self.assertEqual(dns._nameToLabels(b'EXAMPLE.COM'), [b'EXAMPLE', b'COM', b''])