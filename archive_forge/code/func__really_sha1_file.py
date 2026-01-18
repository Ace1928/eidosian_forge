import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def _really_sha1_file(self, abspath, filters):
    if abspath in self._files:
        return sha1(self._files[abspath][0])
    else:
        return None