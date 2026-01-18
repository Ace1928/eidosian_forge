from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase
class FileDescriptorTests(SynchronousTestCase):
    """
    Tests for L{FileDescriptor}.
    """

    def test_writeWithUnicodeRaisesException(self):
        """
        L{FileDescriptor.write} doesn't accept unicode data.
        """
        fileDescriptor = FileDescriptor(reactor=object())
        self.assertRaises(TypeError, fileDescriptor.write, 'foo')

    def test_writeSequenceWithUnicodeRaisesException(self):
        """
        L{FileDescriptor.writeSequence} doesn't accept unicode data.
        """
        fileDescriptor = FileDescriptor(reactor=object())
        self.assertRaises(TypeError, fileDescriptor.writeSequence, [b'foo', 'bar', b'baz'])

    def test_implementInterfaceIPushProducer(self):
        """
        L{FileDescriptor} should implement L{IPushProducer}.
        """
        self.assertTrue(verifyClass(IPushProducer, FileDescriptor))