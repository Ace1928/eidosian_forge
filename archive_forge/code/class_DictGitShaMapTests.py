import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
class DictGitShaMapTests(TestCase, TestGitShaMap):

    def setUp(self):
        TestCase.setUp(self)
        self.cache = DictBzrGitCache()
        self.map = self.cache.idmap