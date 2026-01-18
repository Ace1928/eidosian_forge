import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class InterfaceBaseTestsMixin(NameAndModuleComparisonTestsMixin):

    def _getTargetClass(self):
        raise NotImplementedError

    def _getFallbackClass(self):
        from zope.interface.interface import InterfaceBasePy
        return InterfaceBasePy

    def _makeOne(self, object_should_provide=False, name=None, module=None):

        class IB(self._getTargetClass()):

            def _call_conform(self, conform):
                return conform(self)

            def providedBy(self, obj):
                return object_should_provide
        return IB(name, module)

    def test___call___w___conform___returning_value(self):
        ib = self._makeOne(False)
        conformed = object()

        class _Adapted:

            def __conform__(self, iface):
                return conformed
        self.assertIs(ib(_Adapted()), conformed)

    def test___call___wo___conform___ob_no_provides_w_alternate(self):
        ib = self._makeOne(False)
        __traceback_info__ = (ib, self._getTargetClass())
        adapted = object()
        alternate = object()
        self.assertIs(ib(adapted, alternate), alternate)

    def test___call___w___conform___ob_no_provides_wo_alternate(self):
        ib = self._makeOne(False)
        with self.assertRaises(TypeError) as exc:
            ib(object())
        self.assertIn('Could not adapt', str(exc.exception))

    def test___call___w_no_conform_catches_only_AttributeError(self):
        MissingSomeAttrs.test_raises(self, self._makeOne(), expected_missing='__conform__')