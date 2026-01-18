import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
class ConsoleUITests(TestCase):
    """
    Test cases for L{ConsoleUI}.
    """

    def setUp(self):
        """
        Create a L{ConsoleUI} pointed at a L{FakeFile}.
        """
        self.fakeFile = FakeFile()
        self.ui = ConsoleUI(self.openFile)

    def openFile(self):
        """
        Return the current fake file.
        """
        return self.fakeFile

    def newFile(self, lines):
        """
        Create a new fake file (the next file that self.ui will open) with the
        given list of lines to be returned from readline().
        """
        self.fakeFile = FakeFile()
        self.fakeFile.inlines = lines

    def test_promptYes(self):
        """
        L{ConsoleUI.prompt} writes a message to the console, then reads a line.
        If that line is 'yes', then it returns a L{Deferred} that fires with
        True.
        """
        for okYes in [b'yes', b'Yes', b'yes\n']:
            self.newFile([okYes])
            l = []
            self.ui.prompt('Hello, world!').addCallback(l.append)
            self.assertEqual(['Hello, world!'], self.fakeFile.outchunks)
            self.assertEqual([True], l)
            self.assertTrue(self.fakeFile.closed)

    def test_promptNo(self):
        """
        L{ConsoleUI.prompt} writes a message to the console, then reads a line.
        If that line is 'no', then it returns a L{Deferred} that fires with
        False.
        """
        for okNo in [b'no', b'No', b'no\n']:
            self.newFile([okNo])
            l = []
            self.ui.prompt('Goodbye, world!').addCallback(l.append)
            self.assertEqual(['Goodbye, world!'], self.fakeFile.outchunks)
            self.assertEqual([False], l)
            self.assertTrue(self.fakeFile.closed)

    def test_promptRepeatedly(self):
        """
        L{ConsoleUI.prompt} writes a message to the console, then reads a line.
        If that line is neither 'yes' nor 'no', then it says "Please enter
        'yes' or 'no'" until it gets a 'yes' or a 'no', at which point it
        returns a Deferred that answers either True or False.
        """
        self.newFile([b'what', b'uh', b'okay', b'yes'])
        l = []
        self.ui.prompt(b'Please say something useful.').addCallback(l.append)
        self.assertEqual([True], l)
        self.assertEqual(self.fakeFile.outchunks, [b'Please say something useful.'] + [b"Please type 'yes' or 'no': "] * 3)
        self.assertTrue(self.fakeFile.closed)
        self.newFile([b'blah', b'stuff', b'feh', b'no'])
        l = []
        self.ui.prompt(b'Please say something negative.').addCallback(l.append)
        self.assertEqual([False], l)
        self.assertEqual(self.fakeFile.outchunks, [b'Please say something negative.'] + [b"Please type 'yes' or 'no': "] * 3)
        self.assertTrue(self.fakeFile.closed)

    def test_promptOpenFailed(self):
        """
        If the C{opener} passed to L{ConsoleUI} raises an exception, that
        exception will fail the L{Deferred} returned from L{ConsoleUI.prompt}.
        """

        def raiseIt():
            raise OSError()
        ui = ConsoleUI(raiseIt)
        d = ui.prompt('This is a test.')
        return self.assertFailure(d, IOError)

    def test_warn(self):
        """
        L{ConsoleUI.warn} should output a message to the console object.
        """
        self.ui.warn('Test message.')
        self.assertEqual(['Test message.'], self.fakeFile.outchunks)
        self.assertTrue(self.fakeFile.closed)

    def test_warnOpenFailed(self):
        """
        L{ConsoleUI.warn} should log a traceback if the output can't be opened.
        """

        def raiseIt():
            1 / 0
        ui = ConsoleUI(raiseIt)
        ui.warn('This message never makes it.')
        self.assertEqual(len(self.flushLoggedErrors(ZeroDivisionError)), 1)