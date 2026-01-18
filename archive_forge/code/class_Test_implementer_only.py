import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_implementer_only(Test_classImplementsOnly):

    def _getTargetClass(self):
        from zope.interface.declarations import implementer_only
        return implementer_only

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def _callFUT(self, cls, iface):
        decorator = self._makeOne(iface)
        return decorator(cls)

    def test_function(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decorator = self._makeOne(IFoo)

        def _function():
            raise NotImplementedError()
        self.assertRaises(ValueError, decorator, _function)

    def test_method(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        decorator = self._makeOne(IFoo)

        class Bar:

            def _method(self):
                raise NotImplementedError()
        self.assertRaises(ValueError, decorator, Bar._method)