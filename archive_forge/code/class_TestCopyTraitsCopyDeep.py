import unittest
from traits.api import HasTraits, Instance, Str
class TestCopyTraitsCopyDeep(CopyTraitsBase, TestCopyTraitsSharedCopyNone):
    __test__ = True

    def setUp(self):
        CopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyNone.setUp(self)
        self.baz2.copy_traits(self.baz, copy='deep')