from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
class ServiceInterfaceTests(TestCase):
    """
    Tests for L{twisted.application.service.IService} implementation.
    """

    def setUp(self) -> None:
        """
        Build something that implements IService.
        """
        self.almostService = AlmostService(parent=None, running=False, name=None)

    def test_realService(self) -> None:
        """
        Service implements IService.
        """
        myService = Service()
        verifyObject(IService, myService)

    def test_hasAll(self) -> None:
        """
        AlmostService implements IService.
        """
        verifyObject(IService, self.almostService)

    def test_noName(self) -> None:
        """
        AlmostService with no name does not implement IService.
        """
        self.almostService.makeInvalidByDeletingName()
        with self.assertRaises(BrokenImplementation):
            verifyObject(IService, self.almostService)

    def test_noParent(self) -> None:
        """
        AlmostService with no parent does not implement IService.
        """
        self.almostService.makeInvalidByDeletingParent()
        with self.assertRaises(BrokenImplementation):
            verifyObject(IService, self.almostService)

    def test_noRunning(self) -> None:
        """
        AlmostService with no running does not implement IService.
        """
        self.almostService.makeInvalidByDeletingRunning()
        with self.assertRaises(BrokenImplementation):
            verifyObject(IService, self.almostService)