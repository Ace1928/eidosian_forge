import os
import sys
from breezy import branch, osutils, registry, tests
class TestRegistryIter(tests.TestCase):
    """Test registry iteration behaviors.

    There are dark corner cases here when the registered objects trigger
    addition in the iterated registry.
    """

    def setUp(self):
        super().setUp()
        _registry = registry.Registry()

        def register_more():
            _registry.register('hidden', None)
        self.registry = _registry
        self.registry.register('passive', None)
        self.registry.register('active', register_more)
        self.registry.register('passive-too', None)

        class InvasiveGetter(registry._ObjectGetter):

            def get_obj(inner_self):
                _registry.register('more hidden', None)
                return inner_self._obj
        self.registry.register('hacky', None)
        self.registry._dict['hacky'] = InvasiveGetter(None)

    def _iter_them(self, iter_func_name):
        iter_func = getattr(self.registry, iter_func_name, None)
        self.assertIsNot(None, iter_func)
        count = 0
        for name, func in iter_func():
            count += 1
            self.assertFalse(name in ('hidden', 'more hidden'))
            if func is not None:
                func()
        self.assertEqual(4, count)

    def test_iteritems(self):
        self.assertRaises(RuntimeError, self._iter_them, 'iteritems')

    def test_items(self):
        self._iter_them('items')