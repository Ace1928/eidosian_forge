import os
import platform
import sys
from unittest import skipIf
from dulwich import porcelain
from ..test_porcelain import PorcelainGpgTestCase
from ..utils import build_commit_graph
from .utils import CompatTestCase, run_git_or_fail
@skipIf(platform.python_implementation() == 'PyPy' or sys.platform == 'win32', 'gpgme not easily available or supported on Windows and PyPy')
class TagCreateSignTestCase(PorcelainGpgTestCase, CompatTestCase):

    def setUp(self):
        super().setUp()

    def test_sign(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        cfg = self.repo.get_config()
        cfg.set(('user',), 'signingKey', PorcelainGpgTestCase.DEFAULT_KEY_ID)
        self.import_default_key()
        porcelain.tag_create(self.repo.path, b'tryme', b'foo <foo@bar.com>', b'bar', annotated=True, sign=True)
        run_git_or_fail([f'--git-dir={self.repo.controldir()}', 'tag', '-v', 'tryme'], env={'GNUPGHOME': os.environ['GNUPGHOME']})

    def test_verify(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        self.import_default_key()
        run_git_or_fail([f'--git-dir={self.repo.controldir()}', 'tag', '-u', PorcelainGpgTestCase.DEFAULT_KEY_ID, '-m', 'foo', 'verifyme'], env={'GNUPGHOME': os.environ['GNUPGHOME'], 'GIT_COMMITTER_NAME': 'Joe Example', 'GIT_COMMITTER_EMAIL': 'joe@example.com'})
        tag = self.repo[b'refs/tags/verifyme']
        self.assertNotEqual(tag.signature, None)
        tag.verify()