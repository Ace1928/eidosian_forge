from ....tests import TestCase, TestCaseWithTransport
from ....treebuilder import TreeBuilder
from ..maptree import MapTree, map_file_ids
class MapFileIdTests(TestCase):

    def test_empty(self):
        self.assertEqual({}, map_file_ids(None, [], []))