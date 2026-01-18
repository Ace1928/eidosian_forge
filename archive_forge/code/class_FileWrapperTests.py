from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
class FileWrapperTests(TestCase):
    """
    L{twisted.internet.protocol.FileWrapper}
    """

    def test_write(self):
        """
        L{twisted.internet.protocol.FileWrapper.write}
        """
        wrapper = FileWrapper(BytesIO())
        wrapper.write(b'test1')
        self.assertEqual(wrapper.file.getvalue(), b'test1')
        wrapper = FileWrapper(BytesIO())
        wrapper.write('stuff')
        self.assertNotEqual(wrapper.file.getvalue(), 'stuff')

    def test_writeSequence(self):
        """
        L{twisted.internet.protocol.FileWrapper.writeSequence}
        """
        wrapper = FileWrapper(BytesIO())
        wrapper.writeSequence([b'test1', b'test2'])
        self.assertEqual(wrapper.file.getvalue(), b'test1test2')
        wrapper = FileWrapper(BytesIO())
        self.assertRaises(TypeError, wrapper.writeSequence, ['test3', 'test4'])