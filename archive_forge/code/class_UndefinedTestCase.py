import unittest
from traits.api import HasTraits, Str, Undefined, ReadOnly, Float
class UndefinedTestCase(unittest.TestCase):

    def test_initial_value(self):
        b = Bar()
        self.assertEqual(b.name, Undefined)

    def test_name_change(self):
        b = Bar()
        b.name = 'first'
        self.assertEqual(b.name, 'first')

    def test_read_only_write_once(self):
        f = Foo()
        self.assertEqual(f.name, '')
        self.assertIs(f.original_name, Undefined)
        f.name = 'first'
        self.assertEqual(f.name, 'first')
        self.assertEqual(f.original_name, 'first')
        f.name = 'second'
        self.assertEqual(f.name, 'second')
        self.assertEqual(f.original_name, 'first')

    def test_read_only_write_once_from_constructor(self):
        f = Foo(name='first')
        f.name = 'first'
        self.assertEqual(f.name, 'first')
        self.assertEqual(f.original_name, 'first')
        f.name = 'second'
        self.assertEqual(f.name, 'second')
        self.assertEqual(f.original_name, 'first')