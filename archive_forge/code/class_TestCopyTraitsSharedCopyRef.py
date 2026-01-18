import unittest
from traits.api import HasTraits, Instance, Str
class TestCopyTraitsSharedCopyRef(CopyTraits, CopyTraitsSharedCopyRef):
    __test__ = False

    def setUp(self):
        self.set_shared_copy('ref')