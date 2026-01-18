import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_implementer(Test_classImplements):

    def _getTargetClass(self):
        from zope.interface.declarations import implementer
        return implementer

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def _callFUT(self, cls, *ifaces):
        decorator = self._makeOne(*ifaces)
        return decorator(cls)

    def test_nonclass_cannot_assign_attr(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decorator = self._makeOne(IFoo)
        self.assertRaises(TypeError, decorator, object())

    def test_nonclass_can_assign_attr(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        foo = Foo()
        decorator = self._makeOne(IFoo)
        returned = decorator(foo)
        self.assertTrue(returned is foo)
        spec = foo.__implemented__
        self.assertEqual(spec.__name__, 'zope.interface.tests.test_declarations.?')
        self.assertIsNone(spec.inherit)
        self.assertIs(foo.__implemented__, spec)

    def test_does_not_leak_on_unique_classes(self):
        import gc
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        begin_count = len(gc.get_objects())
        for _ in range(1900):

            class TestClass:
                pass
            self._callFUT(TestClass, IFoo)
        gc.collect()
        end_count = len(gc.get_objects())
        fudge_factor = 0
        self.assertLessEqual(end_count, begin_count + fudge_factor)