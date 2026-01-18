import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class InterfaceBasePyTests(InterfaceBaseTestsMixin, unittest.TestCase):
    _getTargetClass = InterfaceBaseTestsMixin._getFallbackClass

    def test___call___w___conform___miss_ob_provides(self):
        ib = self._makeOne(True)

        class _Adapted:

            def __conform__(self, iface):
                return None
        adapted = _Adapted()
        self.assertIs(ib(adapted), adapted)

    def test___adapt___ob_provides(self):
        ib = self._makeOne(True)
        adapted = object()
        self.assertIs(ib.__adapt__(adapted), adapted)

    def test___adapt___ob_no_provides_uses_hooks(self):
        from zope.interface import interface
        ib = self._makeOne(False)
        adapted = object()
        _missed = []

        def _hook_miss(iface, obj):
            _missed.append((iface, obj))

        def _hook_hit(iface, obj):
            return obj
        with _Monkey(interface, adapter_hooks=[_hook_miss, _hook_hit]):
            self.assertIs(ib.__adapt__(adapted), adapted)
            self.assertEqual(_missed, [(ib, adapted)])