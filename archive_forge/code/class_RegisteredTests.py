import unittest
class RegisteredTests(unittest.TestCase, _ConformsToIRegistrationEvent):

    def _getTargetClass(self):
        from zope.interface.interfaces import Registered
        return Registered

    def test_class_conforms_to_IRegistered(self):
        from zope.interface.interfaces import IRegistered
        from zope.interface.verify import verifyClass
        verifyClass(IRegistered, self._getTargetClass())

    def test_instance_conforms_to_IRegistered(self):
        from zope.interface.interfaces import IRegistered
        from zope.interface.verify import verifyObject
        verifyObject(IRegistered, self._makeOne())