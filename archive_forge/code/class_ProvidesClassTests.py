import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ProvidesClassTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import ProvidesClass
        return ProvidesClass

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_simple_class_one_interface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        spec = self._makeOne(Foo, IFoo)
        self.assertEqual(list(spec), [IFoo])

    def test___reduce__(self):
        from zope.interface.declarations import Provides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        spec = self._makeOne(Foo, IFoo)
        klass, args = spec.__reduce__()
        self.assertIs(klass, Provides)
        self.assertEqual(args, (Foo, IFoo))

    def test___get___class(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        spec = self._makeOne(Foo, IFoo)
        Foo.__provides__ = spec
        self.assertIs(Foo.__provides__, spec)

    def test___get___instance(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        spec = self._makeOne(Foo, IFoo)
        Foo.__provides__ = spec

        def _test():
            foo = Foo()
            return foo.__provides__
        self.assertRaises(AttributeError, _test)