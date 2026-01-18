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
class CompactReprTests(unittest.SynchronousTestCase):
    """
    Tests for L{dns._compactRepr}.
    """
    messageFactory = Foo

    def test_defaults(self):
        """
        L{dns._compactRepr} omits field values and sections which have the
        default value. Flags which are True are always shown.
        """
        self.assertEqual("<Foo alwaysShowField='AS' flags=flagTrue>", repr(self.messageFactory()))

    def test_flagsIfSet(self):
        """
        L{dns._compactRepr} displays flags if they have a non-default value.
        """
        m = self.messageFactory(flagTrue=True, flagFalse=True)
        self.assertEqual("<Foo alwaysShowField='AS' flags=flagTrue,flagFalse>", repr(m))

    def test_nonDefautFields(self):
        """
        L{dns._compactRepr} displays field values if they differ from their
        defaults.
        """
        m = self.messageFactory(field1=10, field2=20)
        self.assertEqual("<Foo field1=10 field2=20 alwaysShowField='AS' flags=flagTrue>", repr(m))

    def test_nonDefaultSections(self):
        """
        L{dns._compactRepr} displays sections which differ from their defaults.
        """
        m = self.messageFactory()
        m.section1 = [1, 1, 1]
        m.section2 = [2, 2, 2]
        self.assertEqual("<Foo alwaysShowField='AS' flags=flagTrue section1=[1, 1, 1] section2=[2, 2, 2]>", repr(m))