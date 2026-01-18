import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class TestClassProvidesRepr(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import ClassProvides
        return ClassProvides

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test__repr__empty(self):
        inst = self._makeOne(type(self), type)
        self.assertEqual(repr(inst), 'directlyProvides(TestClassProvidesRepr)')

    def test__repr__providing_one(self):
        from zope.interface import Interface

        class IFoo(Interface):
            """Does nothing"""
        inst = self._makeOne(type(self), type, IFoo)
        self.assertEqual(repr(inst), 'directlyProvides(TestClassProvidesRepr, IFoo)')

    def test__repr__duplicate_names(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo', __module__='mod1')
        IFoo2 = InterfaceClass('IFoo', __module__='mod2')
        IBaz = InterfaceClass('IBaz')
        inst = self._makeOne(type(self), type, IFoo, IBaz, IFoo2)
        self.assertEqual(repr(inst), 'directlyProvides(TestClassProvidesRepr, IFoo, IBaz, mod2.IFoo)')

    def test__repr__implementedBy(self):
        from zope.interface.declarations import implementedBy
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        inst = implementedBy(Foo)
        self.assertEqual(repr(inst), 'classImplements(Foo, IFoo)')

    def test__repr__implementedBy_generic_callable(self):
        from zope.interface.declarations import implementedBy

        class Callable:

            def __call__(self):
                return self
        inst = implementedBy(Callable())
        self.assertEqual(repr(inst), 'classImplements({}.?)'.format(__name__))
        c = Callable()
        c.__name__ = 'Callable'
        inst = implementedBy(c)
        self.assertEqual(repr(inst), 'classImplements(Callable)')