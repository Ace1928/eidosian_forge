import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
class TdbGitShaMapTests(TestCaseInTempDir, TestGitShaMap):

    def setUp(self):
        TestCaseInTempDir.setUp(self)
        try:
            self.cache = TdbBzrGitCache(os.path.join(self.test_dir, 'foo.tdb'))
        except ModuleNotFoundError:
            raise UnavailableFeature('Missing tdb')
        self.map = self.cache.idmap