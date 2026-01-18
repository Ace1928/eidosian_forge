import os
import stat
from dulwich.objects import Blob, Commit, Tree
from ...revision import Revision
from ...tests import TestCase, TestCaseInTempDir, UnavailableFeature
from ...transport import get_transport
from ..cache import (DictBzrGitCache, IndexBzrGitCache, IndexGitCacheFormat,
def _get_test_commit(self):
    c = Commit()
    c.committer = b'Jelmer <jelmer@samba.org>'
    c.commit_time = 0
    c.commit_timezone = 0
    c.author = b'Jelmer <jelmer@samba.org>'
    c.author_time = 0
    c.author_timezone = 0
    c.message = b'Teh foo bar'
    c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
    return c