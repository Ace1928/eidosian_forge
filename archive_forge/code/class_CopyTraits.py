import unittest
from traits.api import HasTraits, Instance, Str
class CopyTraits(object):

    def test_baz2_s(self):
        self.assertEqual(self.baz2.s, 'baz')
        self.assertEqual(self.baz2.s, self.baz.s)

    def test_baz2_bar_s(self):
        self.assertEqual(self.baz2.bar.s, 'bar')
        self.assertEqual(self.baz2.bar.s, self.baz.bar.s)

    def test_baz2_bar_foo_s(self):
        self.assertEqual(self.baz2.bar.foo.s, 'foo')
        self.assertEqual(self.baz2.bar.foo.s, self.baz.bar.foo.s)

    def test_baz2_shared_s(self):
        self.assertEqual(self.baz2.shared.s, 'shared')
        self.assertEqual(self.baz2.bar.shared.s, 'shared')
        self.assertEqual(self.baz2.bar.foo.shared.s, 'shared')

    def test_baz2_bar(self):
        self.assertIsNot(self.baz2.bar, None)
        self.assertIsNot(self.baz2.bar, self.bar2)
        self.assertIsNot(self.baz2.bar, self.baz.bar)

    def test_baz2_bar_foo(self):
        self.assertIsNot(self.baz2.bar.foo, None)
        self.assertIsNot(self.baz2.bar.foo, self.foo2)
        self.assertIsNot(self.baz2.bar.foo, self.baz.bar.foo)