import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class StubGCVF:

    def __init__(self, canned_get_blocks=None):
        self._group_cache = {}
        self._canned_get_blocks = canned_get_blocks or []

    def _get_blocks(self, read_memos):
        return iter(self._canned_get_blocks)