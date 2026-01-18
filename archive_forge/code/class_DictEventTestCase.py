import unittest
from traits.api import HasTraits, Dict
class DictEventTestCase(unittest.TestCase):

    def test_setitem(self):
        cb = Callback(self, changed={'c': 'cherry'})
        foo = MyClass(cb)
        foo.d['c'] = 'coconut'
        self.assertTrue(cb.called)
        cb = Callback(self, added={'g': 'guava'})
        bar = MyClass(cb)
        bar.d['g'] = 'guava'
        self.assertTrue(cb.called)

    def test_delitem(self):
        cb = Callback(self, removed={'b': 'banana'})
        foo = MyClass(cb)
        del foo.d['b']
        self.assertTrue(cb.called)

    def test_clear(self):
        removed = MyClass(None).d.copy()
        cb = Callback(self, removed=removed)
        foo = MyClass(cb)
        foo.d.clear()
        self.assertTrue(cb.called)

    def test_update(self):
        update_dict = {'a': 'artichoke', 'f': 'fig'}
        cb = Callback(self, changed={'a': 'apple'}, added={'f': 'fig'})
        foo = MyClass(cb)
        foo.d.update(update_dict)
        self.assertTrue(cb.called)

    def test_setdefault(self):
        cb = Callback(self)
        foo = MyClass(cb)
        self.assertEqual(foo.d.setdefault('a', 'dummy'), 'apple')
        self.assertFalse(cb.called)
        cb = Callback(self, added={'f': 'fig'})
        bar = MyClass(cb)
        self.assertTrue(bar.d.setdefault('f', 'fig') == 'fig')
        self.assertTrue(cb.called)

    def test_pop(self):
        cb = Callback(self)
        foo = MyClass(cb)
        self.assertEqual(foo.d.pop('x', 'dummy'), 'dummy')
        self.assertFalse(cb.called)
        cb = Callback(self, removed={'c': 'cherry'})
        bar = MyClass(cb)
        self.assertEqual(bar.d.pop('c'), 'cherry')
        self.assertTrue(cb.called)

    def test_popitem(self):
        foo = MyClass(None)
        foo.d.clear()
        foo.d['x'] = 'xylophone'
        cb = Callback(self, removed={'x': 'xylophone'})
        foo.callback = cb
        self.assertEqual(foo.d.popitem(), ('x', 'xylophone'))
        self.assertTrue(cb.called)

    def test_dynamic_listener(self):
        foo = MyOtherClass()
        func = Callback(self, added={'g': 'guava'})
        foo.on_trait_change(func.__call__, 'd_items')
        foo.d['g'] = 'guava'
        foo.on_trait_change(func.__call__, 'd_items', remove=True)
        self.assertTrue(func.called)
        func2 = Callback(self, removed={'a': 'apple'})
        foo.on_trait_change(func2.__call__, 'd_items')
        del foo.d['a']
        foo.on_trait_change(func2.__call__, 'd_items', remove=True)
        self.assertTrue(func2.called)
        func3 = Callback(self, changed={'b': 'banana'})
        foo.on_trait_change(func3.__call__, 'd_items')
        foo.d['b'] = 'broccoli'
        foo.on_trait_change(func3.__call__, 'd_items', remove=True)
        self.assertTrue(func3.called)