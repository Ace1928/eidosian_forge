import unittest
from traits.api import HasTraits, Instance, Str
class TestCopyTraitsCopyNotSpecifiedSharedRef(CopyTraitsBase, TestCopyTraitsSharedCopyRef):
    __test__ = True

    def setUp(self):
        CopyTraitsBase.setUp(self)
        TestCopyTraitsSharedCopyRef.setUp(self)
        self.baz2.copy_traits(self.baz)