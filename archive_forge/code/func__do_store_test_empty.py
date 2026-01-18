import unittest
from os.path import abspath, dirname, join
import errno
import os
def _do_store_test_empty(self, store):
    store.clear()
    self.assertTrue(store.count() == 0)
    self.assertFalse(store.exists('plop'))
    self.assertRaises(KeyError, lambda: store.get('plop'))
    self.assertTrue(store.put('plop', name='Hello', age=30))
    self.assertTrue(store.exists('plop'))
    self.assertTrue(store.get('plop').get('name') == 'Hello')
    self.assertTrue(store.get('plop').get('age') == 30)
    self.assertTrue(store.count() == 1)
    self.assertTrue('plop' in store.keys())
    store.put('key1', name='Name1', attr1='Common')
    store.put('key2', name='Name2', attr1='Common', attr2='bleh')
    store.put('key3', name='Name3', attr1='Common', attr2='bleh')
    self.assertTrue(store.count() == 4)
    self.assertTrue(store.exists('key1'))
    self.assertTrue(store.exists('key2'))
    self.assertTrue(store.exists('key3'))
    self.assertTrue(len(list(store.find(name='Name2'))) == 1)
    self.assertTrue(list(store.find(name='Name2'))[0][0] == 'key2')
    self.assertTrue(len(list(store.find(attr1='Common'))) == 3)
    self.assertTrue(len(list(store.find(attr2='bleh'))) == 2)
    self.assertTrue(len(list(store.find(attr1='Common', attr2='bleh'))) == 2)
    self.assertTrue(len(list(store.find(name='Name2', attr2='bleh'))) == 1)
    self.assertTrue(len(list(store.find(name='Name1', attr2='bleh'))) == 0)