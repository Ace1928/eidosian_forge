import unittest
class InterfaceClassTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.interface import InterfaceClass
        return InterfaceClass

    def _getTargetInterface(self):
        from zope.interface.interfaces import IInterface
        return IInterface

    def _makeOne(self):
        from zope.interface.interface import Interface
        return Interface

    def test_class_conforms(self):
        from zope.interface.verify import verifyClass
        verifyClass(self._getTargetInterface(), self._getTargetClass())

    def test_instance_conforms(self):
        from zope.interface.verify import verifyObject
        verifyObject(self._getTargetInterface(), self._makeOne())

    def test_instance_consistent__iro__(self):
        from zope.interface import ro
        self.assertTrue(ro.is_consistent(self._getTargetInterface()))

    def test_class_consistent__iro__(self):
        from zope.interface import implementedBy
        from zope.interface import ro
        self.assertTrue(ro.is_consistent(implementedBy(self._getTargetClass())))