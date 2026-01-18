import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestCwdToURL(TestCaseInTempDir):
    """Test that local_path_to_url works based on the cwd"""

    def test_dot(self):
        os.mkdir('mytest')
        os.chdir('mytest')
        url = urlutils.local_path_to_url('.')
        self.assertEndsWith(url, '/mytest')

    def test_non_ascii(self):
        try:
            os.mkdir('dodé')
        except UnicodeError:
            raise TestSkipped('cannot create unicode directory')
        os.chdir('dodé')
        url = urlutils.local_path_to_url('.')
        self.assertEndsWith(url, '/dod%C3%A9')