import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class Test_GCBuildDetails(tests.TestCase):

    def test_acts_like_tuple(self):
        bd = groupcompress._GCBuildDetails((('parent1',), ('parent2',)), ('INDEX', 10, 20, 0, 5))
        self.assertEqual(4, len(bd))
        self.assertEqual(('INDEX', 10, 20, 0, 5), bd[0])
        self.assertEqual(None, bd[1])
        self.assertEqual((('parent1',), ('parent2',)), bd[2])
        self.assertEqual(('group', None), bd[3])

    def test__repr__(self):
        bd = groupcompress._GCBuildDetails((('parent1',), ('parent2',)), ('INDEX', 10, 20, 0, 5))
        self.assertEqual("_GCBuildDetails(('INDEX', 10, 20, 0, 5), (('parent1',), ('parent2',)))", repr(bd))