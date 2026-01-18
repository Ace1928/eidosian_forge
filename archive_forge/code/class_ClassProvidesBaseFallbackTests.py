import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class ClassProvidesBaseFallbackTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import ClassProvidesBaseFallback
        return ClassProvidesBaseFallback

    def _makeOne(self, klass, implements):

        class Derived(self._getTargetClass()):

            def __init__(self, k, i):
                self._cls = k
                self._implements = i
        return Derived(klass, implements)

    def test_w_same_class_via_class(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        cpbp = Foo.__provides__ = self._makeOne(Foo, IFoo)
        self.assertTrue(Foo.__provides__ is cpbp)

    def test_w_same_class_via_instance(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        foo = Foo()
        Foo.__provides__ = self._makeOne(Foo, IFoo)
        self.assertIs(foo.__provides__, IFoo)

    def test_w_different_class(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass

        class Bar(Foo):
            pass
        bar = Bar()
        Foo.__provides__ = self._makeOne(Foo, IFoo)
        self.assertRaises(AttributeError, getattr, Bar, '__provides__')
        self.assertRaises(AttributeError, getattr, bar, '__provides__')