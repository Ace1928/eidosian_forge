import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
class TestGitSDist(base.BaseTestCase):

    def setUp(self):
        super(TestGitSDist, self).setUp()
        stdout, _, return_code = self._run_cmd('git', ('init',))
        if return_code:
            self.skipTest('git not installed')
        stdout, _, return_code = self._run_cmd('git', ('add', '.'))
        stdout, _, return_code = self._run_cmd('git', ('commit', '-m', 'Turn this into a git repo'))
        stdout, _, return_code = self.run_setup('sdist', '--formats=gztar')

    def test_sdist_git_extra_files(self):
        """Test that extra files found in git are correctly added."""
        tf_path = glob.glob(os.path.join('dist', '*.tar.gz'))[0]
        tf = tarfile.open(tf_path)
        names = ['/'.join(p.split('/')[1:]) for p in tf.getnames()]
        self.assertIn('git-extra-file.txt', names)