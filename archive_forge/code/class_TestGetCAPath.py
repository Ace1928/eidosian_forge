import os
import sys
from ..transport.http import ca_bundle
from . import TestCaseInTempDir, TestSkipped
class TestGetCAPath(TestCaseInTempDir):

    def setUp(self):
        super().setUp()
        self.overrideEnv('CURL_CA_BUNDLE', None)
        self.overrideEnv('PATH', None)

    def _make_file(self, in_dir='.'):
        fname = os.path.join(in_dir, 'curl-ca-bundle.crt')
        with open(fname, 'w') as f:
            f.write('spam')

    def test_found_nothing(self):
        self.assertEqual('', ca_bundle.get_ca_path(use_cache=False))

    def test_env_var(self):
        self.overrideEnv('CURL_CA_BUNDLE', 'foo.bar')
        self._make_file()
        self.assertEqual('foo.bar', ca_bundle.get_ca_path(use_cache=False))

    def test_in_path(self):
        if sys.platform != 'win32':
            raise TestSkipped('Searching in PATH implemented only for win32')
        os.mkdir('foo')
        in_dir = os.path.join(self.test_dir, 'foo')
        self._make_file(in_dir=in_dir)
        self.overrideEnv('PATH', in_dir)
        shouldbe = os.path.join(in_dir, 'curl-ca-bundle.crt')
        self.assertEqual(shouldbe, ca_bundle.get_ca_path(use_cache=False))