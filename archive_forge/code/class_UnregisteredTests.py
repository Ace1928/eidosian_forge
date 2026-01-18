import unittest
class UnregisteredTests(unittest.TestCase, _ConformsToIRegistrationEvent):

    def _getTargetClass(self):
        from zope.interface.interfaces import Unregistered
        return Unregistered

    def test_class_conforms_to_IUnregistered(self):
        from zope.interface.interfaces import IUnregistered
        from zope.interface.verify import verifyClass
        verifyClass(IUnregistered, self._getTargetClass())

    def test_instance_conforms_to_IUnregistered(self):
        from zope.interface.interfaces import IUnregistered
        from zope.interface.verify import verifyObject
        verifyObject(IUnregistered, self._makeOne())