import unittest
from traits.api import HasTraits, Instance, Str
class CopyTraitsSharedCopyNone(object):

    def test_baz2_shared(self):
        self.assertIsNot(self.baz2.shared, None)
        self.assertIsNot(self.baz2.shared, self.shared2)
        self.assertIsNot(self.baz2.shared, self.shared)

    def test_baz2_bar_shared(self):
        self.assertIsNot(self.baz2.bar.shared, None)
        self.assertIsNot(self.baz2.bar.shared, self.shared2)
        self.assertIsNot(self.baz2.bar.shared, self.shared)
        self.assertIsNot(self.baz2.bar.shared, self.baz2.shared)

    def test_baz2_bar_foo_shared(self):
        self.assertIsNot(self.baz2.bar.foo.shared, None)
        self.assertIsNot(self.baz2.bar.foo.shared, self.shared2)
        self.assertIsNot(self.baz2.bar.foo.shared, self.shared)
        self.assertIsNot(self.baz2.bar.foo.shared, self.baz2.shared)

    def test_baz2_bar_and_foo_shared(self):
        self.assertIs(self.baz2.bar.shared, self.baz2.bar.foo.shared)
        self.assertIsNot(self.baz2.shared, self.baz2.bar.foo.shared)