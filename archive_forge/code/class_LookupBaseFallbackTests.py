import unittest
from zope.interface.tests import OptimizationTestMixin
class LookupBaseFallbackTests(unittest.TestCase):

    def _getFallbackClass(self):
        from zope.interface.adapter import LookupBaseFallback
        return LookupBaseFallback
    _getTargetClass = _getFallbackClass

    def _makeOne(self, uc_lookup=None, uc_lookupAll=None, uc_subscriptions=None):
        if uc_lookup is None:

            def uc_lookup(self, required, provided, name):
                pass
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
        return Derived()

    def test_lookup_w_invalid_name(self):

        def _lookup(self, required, provided, name):
            self.fail('This should never be called')
        lb = self._makeOne(uc_lookup=_lookup)
        with self.assertRaises(ValueError):
            lb.lookup(('A',), 'B', object())

    def test_lookup_miss_no_default(self):
        _called_with = []

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A',), 'B', 'C')
        self.assertIsNone(found)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])

    def test_lookup_miss_w_default(self):
        _called_with = []
        _default = object()

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A',), 'B', 'C', _default)
        self.assertIs(found, _default)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])

    def test_lookup_not_cached(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A',), 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])
        self.assertEqual(_results, [b, c])

    def test_lookup_cached(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A',), 'B', 'C')
        found = lb.lookup(('A',), 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])
        self.assertEqual(_results, [b, c])

    def test_lookup_not_cached_multi_required(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A', 'D'), 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A', 'D'), 'B', 'C')])
        self.assertEqual(_results, [b, c])

    def test_lookup_cached_multi_required(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A', 'D'), 'B', 'C')
        found = lb.lookup(('A', 'D'), 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A', 'D'), 'B', 'C')])
        self.assertEqual(_results, [b, c])

    def test_lookup_not_cached_after_changed(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup(('A',), 'B', 'C')
        lb.changed(lb)
        found = lb.lookup(('A',), 'B', 'C')
        self.assertIs(found, b)
        self.assertEqual(_called_with, [(('A',), 'B', 'C'), (('A',), 'B', 'C')])
        self.assertEqual(_results, [c])

    def test_lookup1_w_invalid_name(self):

        def _lookup(self, required, provided, name):
            self.fail('This should never be called')
        lb = self._makeOne(uc_lookup=_lookup)
        with self.assertRaises(ValueError):
            lb.lookup1('A', 'B', object())

    def test_lookup1_miss_no_default(self):
        _called_with = []

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C')
        self.assertIsNone(found)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])

    def test_lookup1_miss_w_default(self):
        _called_with = []
        _default = object()

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C', _default)
        self.assertIs(found, _default)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])

    def test_lookup1_miss_w_default_negative_cache(self):
        _called_with = []
        _default = object()

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C', _default)
        self.assertIs(found, _default)
        found = lb.lookup1('A', 'B', 'C', _default)
        self.assertIs(found, _default)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])

    def test_lookup1_not_cached(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])
        self.assertEqual(_results, [b, c])

    def test_lookup1_cached(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C')
        found = lb.lookup1('A', 'B', 'C')
        self.assertIs(found, a)
        self.assertEqual(_called_with, [(('A',), 'B', 'C')])
        self.assertEqual(_results, [b, c])

    def test_lookup1_not_cached_after_changed(self):
        _called_with = []
        a, b, c = (object(), object(), object())
        _results = [a, b, c]

        def _lookup(self, required, provided, name):
            _called_with.append((required, provided, name))
            return _results.pop(0)
        lb = self._makeOne(uc_lookup=_lookup)
        found = lb.lookup1('A', 'B', 'C')
        lb.changed(lb)
        found = lb.lookup1('A', 'B', 'C')
        self.assertIs(found, b)
        self.assertEqual(_called_with, [(('A',), 'B', 'C'), (('A',), 'B', 'C')])
        self.assertEqual(_results, [c])

    def test_adapter_hook_w_invalid_name(self):
        req, prv = (object(), object())
        lb = self._makeOne()
        with self.assertRaises(ValueError):
            lb.adapter_hook(prv, req, object())

    def test_adapter_hook_miss_no_default(self):
        req, prv = (object(), object())
        lb = self._makeOne()
        found = lb.adapter_hook(prv, req, '')
        self.assertIsNone(found)

    def test_adapter_hook_miss_w_default(self):
        req, prv, _default = (object(), object(), object())
        lb = self._makeOne()
        found = lb.adapter_hook(prv, req, '', _default)
        self.assertIs(found, _default)

    def test_adapter_hook_hit_factory_returns_None(self):
        _f_called_with = []

        def _factory(context):
            _f_called_with.append(context)

        def _lookup(self, required, provided, name):
            return _factory
        req, prv, _default = (object(), object(), object())
        lb = self._makeOne(uc_lookup=_lookup)
        adapted = lb.adapter_hook(prv, req, 'C', _default)
        self.assertIs(adapted, _default)
        self.assertEqual(_f_called_with, [req])

    def test_adapter_hook_hit_factory_returns_adapter(self):
        _f_called_with = []
        _adapter = object()

        def _factory(context):
            _f_called_with.append(context)
            return _adapter

        def _lookup(self, required, provided, name):
            return _factory
        req, prv, _default = (object(), object(), object())
        lb = self._makeOne(uc_lookup=_lookup)
        adapted = lb.adapter_hook(prv, req, 'C', _default)
        self.assertIs(adapted, _adapter)
        self.assertEqual(_f_called_with, [req])

    def test_adapter_hook_super_unwraps(self):
        _f_called_with = []

        def _factory(context):
            _f_called_with.append(context)
            return context

        def _lookup(self, required, provided, name=''):
            return _factory
        required = super()
        provided = object()
        lb = self._makeOne(uc_lookup=_lookup)
        adapted = lb.adapter_hook(provided, required)
        self.assertIs(adapted, self)
        self.assertEqual(_f_called_with, [self])

    def test_queryAdapter(self):
        _f_called_with = []
        _adapter = object()

        def _factory(context):
            _f_called_with.append(context)
            return _adapter

        def _lookup(self, required, provided, name):
            return _factory
        req, prv, _default = (object(), object(), object())
        lb = self._makeOne(uc_lookup=_lookup)
        adapted = lb.queryAdapter(req, prv, 'C', _default)
        self.assertIs(adapted, _adapter)
        self.assertEqual(_f_called_with, [req])

    def test_lookupAll_uncached(self):
        _called_with = []
        _results = [object(), object(), object()]

        def _lookupAll(self, required, provided):
            _called_with.append((required, provided))
            return tuple(_results)
        lb = self._makeOne(uc_lookupAll=_lookupAll)
        found = lb.lookupAll('A', 'B')
        self.assertEqual(found, tuple(_results))
        self.assertEqual(_called_with, [(('A',), 'B')])

    def test_lookupAll_cached(self):
        _called_with = []
        _results = [object(), object(), object()]

        def _lookupAll(self, required, provided):
            _called_with.append((required, provided))
            return tuple(_results)
        lb = self._makeOne(uc_lookupAll=_lookupAll)
        found = lb.lookupAll('A', 'B')
        found = lb.lookupAll('A', 'B')
        self.assertEqual(found, tuple(_results))
        self.assertEqual(_called_with, [(('A',), 'B')])

    def test_subscriptions_uncached(self):
        _called_with = []
        _results = [object(), object(), object()]

        def _subscriptions(self, required, provided):
            _called_with.append((required, provided))
            return tuple(_results)
        lb = self._makeOne(uc_subscriptions=_subscriptions)
        found = lb.subscriptions('A', 'B')
        self.assertEqual(found, tuple(_results))
        self.assertEqual(_called_with, [(('A',), 'B')])

    def test_subscriptions_cached(self):
        _called_with = []
        _results = [object(), object(), object()]

        def _subscriptions(self, required, provided):
            _called_with.append((required, provided))
            return tuple(_results)
        lb = self._makeOne(uc_subscriptions=_subscriptions)
        found = lb.subscriptions('A', 'B')
        found = lb.subscriptions('A', 'B')
        self.assertEqual(found, tuple(_results))
        self.assertEqual(_called_with, [(('A',), 'B')])