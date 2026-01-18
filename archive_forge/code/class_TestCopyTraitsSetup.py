import unittest
from traits.api import HasTraits, Instance, Str
class TestCopyTraitsSetup(CopyTraitsBase):
    __test__ = True

    def setUp(self):
        super().setUp()

    def test_setup(self):
        self.assertIs(self.foo, self.bar.foo)
        self.assertIs(self.bar, self.baz.bar)
        self.assertIs(self.foo.shared, self.shared)
        self.assertIs(self.bar.shared, self.shared)
        self.assertIs(self.baz.shared, self.shared)
        self.assertIs(self.foo2, self.bar2.foo)
        self.assertIs(self.bar2, self.baz2.bar)
        self.assertIs(self.foo2.shared, self.shared2)
        self.assertIs(self.bar2.shared, self.shared2)
        self.assertIs(self.baz2.shared, self.shared2)