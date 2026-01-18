import shutil
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import pytest
from IPython.core.profileapp import list_bundled_profiles, list_profiles_in
from IPython.core.profiledir import ProfileDir
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.process import getoutput
class ProfileStartupTest(TestCase):

    def setUp(self):
        self.pd = ProfileDir.create_profile_dir_by_name(IP_TEST_DIR, 'test')
        self.options = ['--ipython-dir', IP_TEST_DIR, '--profile', 'test']
        self.fname = TMP_TEST_DIR / 'test.py'

    def tearDown(self):
        shutil.rmtree(self.pd.location)

    def init(self, startup_file, startup, test):
        with open(Path(self.pd.startup_dir) / startup_file, 'w', encoding='utf-8') as f:
            f.write(startup)
        with open(self.fname, 'w', encoding='utf-8') as f:
            f.write(test)

    def validate(self, output):
        tt.ipexec_validate(self.fname, output, '', options=self.options)

    def test_startup_py(self):
        self.init('00-start.py', 'zzz=123\n', 'print(zzz)\n')
        self.validate('123')

    def test_startup_ipy(self):
        self.init('00-start.ipy', '%xmode plain\n', '')
        self.validate('Exception reporting mode: Plain')