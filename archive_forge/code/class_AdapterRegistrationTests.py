import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class AdapterRegistrationTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.registry import AdapterRegistration
        return AdapterRegistration

    def _makeOne(self, component=None):
        from zope.interface.declarations import InterfaceClass

        class IFoo(InterfaceClass):
            pass
        ifoo = IFoo('IFoo')
        ibar = IFoo('IBar')

        class _Registry:

            def __repr__(self):
                return '_REGISTRY'
        registry = _Registry()
        name = 'name'
        doc = 'DOCSTRING'
        klass = self._getTargetClass()
        return (klass(registry, (ibar,), ifoo, name, component, doc), registry, name)

    def test_class_conforms_to_IAdapterRegistration(self):
        from zope.interface.interfaces import IAdapterRegistration
        from zope.interface.verify import verifyClass
        verifyClass(IAdapterRegistration, self._getTargetClass())

    def test_instance_conforms_to_IAdapterRegistration(self):
        from zope.interface.interfaces import IAdapterRegistration
        from zope.interface.verify import verifyObject
        ar, _, _ = self._makeOne()
        verifyObject(IAdapterRegistration, ar)

    def test___repr__(self):

        class _Component:
            __name__ = 'TEST'
        _component = _Component()
        ar, _registry, _name = self._makeOne(_component)
        self.assertEqual(repr(ar), ('AdapterRegistration(_REGISTRY, [IBar], IFoo, %r, TEST, ' + "'DOCSTRING')") % _name)

    def test___repr___provided_wo_name(self):

        class _Component:

            def __repr__(self):
                return 'TEST'
        _component = _Component()
        ar, _registry, _name = self._makeOne(_component)
        ar.provided = object()
        self.assertEqual(repr(ar), ('AdapterRegistration(_REGISTRY, [IBar], None, %r, TEST, ' + "'DOCSTRING')") % _name)

    def test___repr___component_wo_name(self):

        class _Component:

            def __repr__(self):
                return 'TEST'
        _component = _Component()
        ar, _registry, _name = self._makeOne(_component)
        ar.provided = object()
        self.assertEqual(repr(ar), ('AdapterRegistration(_REGISTRY, [IBar], None, %r, TEST, ' + "'DOCSTRING')") % _name)

    def test___hash__(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertEqual(ar.__hash__(), id(ar))

    def test___eq___identity(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertTrue(ar == ar)

    def test___eq___hit(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        self.assertTrue(ar == ar2)

    def test___eq___miss(self):
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        self.assertFalse(ar == ar2)

    def test___ne___identity(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertFalse(ar != ar)

    def test___ne___miss(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        self.assertFalse(ar != ar2)

    def test___ne___hit_component(self):
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        self.assertTrue(ar != ar2)

    def test___ne___hit_provided(self):
        from zope.interface.declarations import InterfaceClass

        class IFoo(InterfaceClass):
            pass
        ibaz = IFoo('IBaz')
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        ar2.provided = ibaz
        self.assertTrue(ar != ar2)

    def test___ne___hit_required(self):
        from zope.interface.declarations import InterfaceClass

        class IFoo(InterfaceClass):
            pass
        ibaz = IFoo('IBaz')
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        ar2.required = (ibaz,)
        self.assertTrue(ar != ar2)

    def test___lt___identity(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertFalse(ar < ar)

    def test___lt___hit(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        self.assertFalse(ar < ar2)

    def test___lt___miss(self):
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        ar2.name = _name + '2'
        self.assertTrue(ar < ar2)

    def test___le___identity(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertTrue(ar <= ar)

    def test___le___hit(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        self.assertTrue(ar <= ar2)

    def test___le___miss(self):
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        ar2.name = _name + '2'
        self.assertTrue(ar <= ar2)

    def test___gt___identity(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertFalse(ar > ar)

    def test___gt___hit(self):
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        ar2.name = _name + '2'
        self.assertTrue(ar2 > ar)

    def test___gt___miss(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        self.assertFalse(ar2 > ar)

    def test___ge___identity(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        self.assertTrue(ar >= ar)

    def test___ge___miss(self):
        _component = object()
        _component2 = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component2)
        ar2.name = _name + '2'
        self.assertFalse(ar >= ar2)

    def test___ge___hit(self):
        _component = object()
        ar, _registry, _name = self._makeOne(_component)
        ar2, _, _ = self._makeOne(_component)
        ar2.name = _name + '2'
        self.assertTrue(ar2 >= ar)