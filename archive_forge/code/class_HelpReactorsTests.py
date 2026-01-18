import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
class HelpReactorsTests(TestCase):
    """
    --help-reactors lists the available reactors
    """

    def setUp(self):
        """
        Get the text from --help-reactors
        """
        self.options = app.ReactorSelectionMixin()
        self.options.messageOutput = StringIO()
        self.assertRaises(SystemExit, self.options.opt_help_reactors)
        self.message = self.options.messageOutput.getvalue()

    @skipIf(asyncio, 'Not applicable, asyncio is available')
    def test_lacksAsyncIO(self):
        """
        --help-reactors should NOT display the asyncio reactor on Python < 3.4
        """
        self.assertIn(twisted_reactors.asyncio.description, self.message)
        self.assertIn('!' + twisted_reactors.asyncio.shortName, self.message)

    @skipIf(not asyncio, 'asyncio library not available')
    def test_hasAsyncIO(self):
        """
        --help-reactors should display the asyncio reactor on Python >= 3.4
        """
        self.assertIn(twisted_reactors.asyncio.description, self.message)
        self.assertNotIn('!' + twisted_reactors.asyncio.shortName, self.message)

    @skipIf(platformType != 'win32', 'Test only applicable on Windows')
    def test_iocpWin32(self):
        """
        --help-reactors should display the iocp reactor on Windows
        """
        self.assertIn(twisted_reactors.iocp.description, self.message)
        self.assertNotIn('!' + twisted_reactors.iocp.shortName, self.message)

    @skipIf(platformType == 'win32', 'Test not applicable on Windows')
    def test_iocpNotWin32(self):
        """
        --help-reactors should NOT display the iocp reactor on Windows
        """
        self.assertIn(twisted_reactors.iocp.description, self.message)
        self.assertIn('!' + twisted_reactors.iocp.shortName, self.message)

    def test_onlySupportedReactors(self):
        """
        --help-reactors with only supported reactors
        """

        def getReactorTypes():
            yield twisted_reactors.default
        options = app.ReactorSelectionMixin()
        options._getReactorTypes = getReactorTypes
        options.messageOutput = StringIO()
        self.assertRaises(SystemExit, options.opt_help_reactors)
        message = options.messageOutput.getvalue()
        self.assertNotIn('reactors not available', message)