import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_classImplementsOnly(_ImplementsTestMixin, unittest.TestCase):
    FUT_SETS_PROVIDED_BY = False

    def _callFUT(self, cls, iface):
        from zope.interface.declarations import classImplementsOnly
        classImplementsOnly(cls, iface)
        return cls

    def test_w_existing_Implements(self):
        from zope.interface.declarations import Implements
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar')
        impl = Implements(IFoo)
        impl.declared = (IFoo,)

        class Foo:
            __implemented__ = impl
        impl.inherit = Foo
        self._callFUT(Foo, IBar)
        self.assertTrue(Foo.__implemented__ is impl)
        self.assertEqual(impl.inherit, None)
        self.assertEqual(impl.declared, (IBar,))

    def test_class(self):
        from zope.interface.declarations import Implements
        from zope.interface.interface import InterfaceClass
        IBar = InterfaceClass('IBar')
        old_spec = Implements(IBar)

        class Foo:
            __implemented__ = old_spec
        self._check_implementer(Foo, old_spec, '?', inherit=None)

    def test_redundant_with_super_still_implements(self):
        Base, IBase = self._check_implementer(type('Foo', (object,), {}), inherit=None)

        class Child(Base):
            pass
        self._callFUT(Child, IBase)
        self.assertTrue(IBase.implementedBy(Child))