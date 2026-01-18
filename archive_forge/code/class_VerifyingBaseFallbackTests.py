import unittest
from zope.interface.tests import OptimizationTestMixin
class VerifyingBaseFallbackTests(unittest.TestCase):

    def _getFallbackClass(self):
        from zope.interface.adapter import VerifyingBaseFallback
        return VerifyingBaseFallback
    _getTargetClass = _getFallbackClass

    def _makeOne(self, registry, uc_lookup=None, uc_lookupAll=None, uc_subscriptions=None):
        if uc_lookup is None:

            def uc_lookup(self, required, provided, name):
                raise NotImplementedError()
        if uc_lookupAll is None:

            def uc_lookupAll(self, required, provided):
                raise NotImplementedError()
        if uc_subscriptions is None:

            def uc_subscriptions(self, required, provided):
                raise NotImplementedError()

        class Derived(self._getTargetClass()):
            _uncached_lookup = uc_lookup
            _uncached_lookupAll = uc_lookupAll
            _uncached_subscriptions = uc_subscriptions

            def __init__(self, registry):
                super().__init__()
                self._registry = registry
        derived = Derived(registry)
        derived.changed(derived)
        return derived

    def _makeRegistry(self, depth):

        class WithGeneration:
            _generation = 1

        class Registry:

            def __init__(self, depth):
                self.ro = [WithGeneration() for i in range(depth)]
        return Registry(depth)

    def test_lookup(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        reg = self._makeRegistry(3)
        lb = self._makeOne(reg, uc_lookup=_lookup)
        found = lb.lookup(('A',), 'B', 'C')
        found = lb.lookup(('A',), 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])
        self.assertEqual(_results, [b, c])
        reg.ro[1]._generation += 1
        found = lb.lookup(('A',), 'B', 'C')
        self.assertIs(found, b)
        self.assertEqual(_called_with, [(('A',), 'B', 'C'), (('A',), 'B', 'C')])
        self.assertEqual(_results, [c])

    def test_lookup1(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        reg = self._makeRegistry(3)
        lb = self._makeOne(reg, uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C')
        found = lb.lookup1('A', 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])
        self.assertEqual(_results, [b, c])
        reg.ro[1]._generation += 1
        found = lb.lookup1('A', 'B', 'C')
        self.assertIs(found, b)
        self.assertEqual(_called_with, [(('A',), 'B', 'C'), (('A',), 'B', 'C')])
        self.assertEqual(_results, [c])

    def test_adapter_hook(self):
        a, b, _c = [object(), object(), object()]

        def _factory1(context):
            return a

        def _factory2(context):
            return b

        def _factory3(context):
            self.fail('This should never be called')
        _factories = [_factory1, _factory2, _factory3]

        def _lookup(self, required, provided, name):
            return _factories.pop(0)
        req, prv, _default = (object(), object(), object())
        reg = self._makeRegistry(3)
        lb = self._makeOne(reg, uc_lookup=_lookup)
        adapted = lb.adapter_hook(prv, req, 'C', _default)
        self.assertIs(adapted, a)
        adapted = lb.adapter_hook(prv, req, 'C', _default)
        self.assertIs(adapted, a)
        reg.ro[1]._generation += 1
        adapted = lb.adapter_hook(prv, req, 'C', _default)
        self.assertIs(adapted, b)

    def test_queryAdapter(self):
        a, b, _c = [object(), object(), object()]

        def _factory1(context):
            return a

        def _factory2(context):
            return b

        def _factory3(context):
            self.fail('This should never be called')
        _factories = [_factory1, _factory2, _factory3]

        def _lookup(self, required, provided, name):
            return _factories.pop(0)
        req, prv, _default = (object(), object(), object())
        reg = self._makeRegistry(3)
        lb = self._makeOne(reg, uc_lookup=_lookup)
        adapted = lb.queryAdapter(req, prv, 'C', _default)
        self.assertIs(adapted, a)
        adapted = lb.queryAdapter(req, prv, 'C', _default)
        self.assertIs(adapted, a)
        reg.ro[1]._generation += 1
        adapted = lb.adapter_hook(prv, req, 'C', _default)
        self.assertIs(adapted, b)

    def test_lookupAll(self):
        _results_1 = [object(), object(), object()]
        _results_2 = [object(), object(), object()]
        _results = [_results_1, _results_2]

        def _lookupAll(self, required, provided):
            return tuple(_results.pop(0))
        reg = self._makeRegistry(3)
        lb = self._makeOne(reg, uc_lookupAll=_lookupAll)
        found = lb.lookupAll('A', 'B')
        self.assertEqual(found, tuple(_results_1))
        found = lb.lookupAll('A', 'B')
        self.assertEqual(found, tuple(_results_1))
        reg.ro[1]._generation += 1
        found = lb.lookupAll('A', 'B')
        self.assertEqual(found, tuple(_results_2))

    def test_subscriptions(self):
        _results_1 = [object(), object(), object()]
        _results_2 = [object(), object(), object()]
        _results = [_results_1, _results_2]

        def _subscriptions(self, required, provided):
            return tuple(_results.pop(0))
        reg = self._makeRegistry(3)
        lb = self._makeOne(reg, uc_subscriptions=_subscriptions)
        found = lb.subscriptions('A', 'B')
        self.assertEqual(found, tuple(_results_1))
        found = lb.subscriptions('A', 'B')
        self.assertEqual(found, tuple(_results_1))
        reg.ro[1]._generation += 1
        found = lb.subscriptions('A', 'B')
        self.assertEqual(found, tuple(_results_2))