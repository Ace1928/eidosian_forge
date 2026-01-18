from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
class ApplicationTests(TestCase):
    """
    Tests for L{twisted.application.service.Application}.
    """

    def test_applicationComponents(self) -> None:
        """
        Check L{twisted.application.service.Application} instantiation.
        """
        app = Application('app-name')
        self.assertTrue(verifyObject(IService, IService(app)))
        self.assertTrue(verifyObject(IServiceCollection, IServiceCollection(app)))
        self.assertTrue(verifyObject(IProcess, IProcess(app)))
        self.assertTrue(verifyObject(IPersistable, IPersistable(app)))