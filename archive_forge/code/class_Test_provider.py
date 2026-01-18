import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_provider(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import provider
        return provider

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_w_class(self):
        from zope.interface.declarations import ClassProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @self._makeOne(IFoo)
        class Foo:
            pass
        self.assertIsInstance(Foo.__provides__, ClassProvides)
        self.assertEqual(list(Foo.__provides__), [IFoo])