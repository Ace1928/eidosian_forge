import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class UtilityRegistrationTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.registry import UtilityRegistration
        return UtilityRegistration

    def _makeOne(self, component=None, factory=None):
        from zope.interface.declarations import InterfaceClass

        class InterfaceClassSubclass(InterfaceClass):
            pass
        ifoo = InterfaceClassSubclass('IFoo')

        class _Registry:

            def __repr__(self):
                return '_REGISTRY'
        registry = _Registry()
        name = 'name'
        doc = 'DOCSTRING'
        klass = self._getTargetClass()
        return (klass(registry, ifoo, name, component, doc, factory), registry, name)

    def test_class_conforms_to_IUtilityRegistration(self):
        from zope.interface.interfaces import IUtilityRegistration
        from zope.interface.verify import verifyClass
        verifyClass(IUtilityRegistration, self._getTargetClass())

    def test_instance_conforms_to_IUtilityRegistration(self):
        from zope.interface.interfaces import IUtilityRegistration
        from zope.interface.verify import verifyObject
        ur, _, _ = self._makeOne()
        verifyObject(IUtilityRegistration, ur)

    def test___repr__(self):

        class _Component:
            __name__ = 'TEST'
        _component = _Component()
        ur, _registry, _name = self._makeOne(_component)
        self.assertEqual(repr(ur), "UtilityRegistration(_REGISTRY, IFoo, %r, TEST, None, 'DOCSTRING')" % _name)

    def test___repr___provided_wo_name(self):

        class _Component:

            def __repr__(self):
                return 'TEST'
        _component = _Component()
        ur, _registry, _name = self._makeOne(_component)
        ur.provided = object()
        self.assertEqual(repr(ur), "UtilityRegistration(_REGISTRY, None, %r, TEST, None, 'DOCSTRING')" % _name)

    def test___repr___component_wo_name(self):

        class _Component:

            def __repr__(self):
                return 'TEST'
        _component = _Component()
        ur, _registry, _name = self._makeOne(_component)
        ur.provided = object()
        self.assertEqual(repr(ur), "UtilityRegistration(_REGISTRY, None, %r, TEST, None, 'DOCSTRING')" % _name)

    def test___hash__(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertEqual(ur.__hash__(), id(ur))

    def test___eq___identity(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertTrue(ur == ur)

    def test___eq___hit(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component)
        self.assertTrue(ur == ur2)

    def test___eq___miss(self):
        _component = object()
        _component2 = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component2)
        self.assertFalse(ur == ur2)

    def test___ne___identity(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertFalse(ur != ur)

    def test___ne___hit(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component)
        self.assertFalse(ur != ur2)

    def test___ne___miss(self):
        _component = object()
        _component2 = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component2)
        self.assertTrue(ur != ur2)

    def test___lt___identity(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertFalse(ur < ur)

    def test___lt___hit(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component)
        self.assertFalse(ur < ur2)

    def test___lt___miss(self):
        _component = object()
        _component2 = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component2)
        ur2.name = _name + '2'
        self.assertTrue(ur < ur2)

    def test___le___identity(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertTrue(ur <= ur)

    def test___le___hit(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component)
        self.assertTrue(ur <= ur2)

    def test___le___miss(self):
        _component = object()
        _component2 = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component2)
        ur2.name = _name + '2'
        self.assertTrue(ur <= ur2)

    def test___gt___identity(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertFalse(ur > ur)

    def test___gt___hit(self):
        _component = object()
        _component2 = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component2)
        ur2.name = _name + '2'
        self.assertTrue(ur2 > ur)

    def test___gt___miss(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component)
        self.assertFalse(ur2 > ur)

    def test___ge___identity(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        self.assertTrue(ur >= ur)

    def test___ge___miss(self):
        _component = object()
        _component2 = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component2)
        ur2.name = _name + '2'
        self.assertFalse(ur >= ur2)

    def test___ge___hit(self):
        _component = object()
        ur, _registry, _name = self._makeOne(_component)
        ur2, _, _ = self._makeOne(_component)
        ur2.name = _name + '2'
        self.assertTrue(ur2 >= ur)