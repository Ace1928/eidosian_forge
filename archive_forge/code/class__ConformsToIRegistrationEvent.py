import unittest
class _ConformsToIRegistrationEvent(_ConformsToIObjectEvent):

    def test_class_conforms_to_IRegistrationEvent(self):
        from zope.interface.interfaces import IRegistrationEvent
        from zope.interface.verify import verifyClass
        verifyClass(IRegistrationEvent, self._getTargetClass())

    def test_instance_conforms_to_IRegistrationEvent(self):
        from zope.interface.interfaces import IRegistrationEvent
        from zope.interface.verify import verifyObject
        verifyObject(IRegistrationEvent, self._makeOne())