import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
class TestVersionBzrLogLocation(TestCaseInTempDir):

    def default_log(self):
        return os.path.join(os.environ['BRZ_HOME'], 'breezy', 'brz.log')

    def test_simple(self):
        brz_log = 'my.brz.log'
        self.overrideEnv('BRZ_LOG', brz_log)
        self.assertPathDoesNotExist([self.default_log(), brz_log])
        out = self.run_brz_subprocess('version')[0]
        self.assertTrue(len(out) > 0)
        self.assertContainsRe(out, b'(?m)^  Breezy log file: ' + brz_log.encode('ascii'))
        self.assertPathExists(brz_log)
        self.assertPathDoesNotExist(self.default_log())

    def test_dev_null(self):
        if sys.platform == 'win32':
            brz_log = 'NUL'
        else:
            brz_log = '/dev/null'
        self.overrideEnv('BRZ_LOG', brz_log)
        self.assertPathDoesNotExist(self.default_log())
        out = self.run_brz_subprocess('version')[0]
        self.assertTrue(len(out) > 0)
        self.assertContainsRe(out, b'(?m)^  Breezy log file: ' + brz_log.encode('ascii'))
        self.assertPathDoesNotExist(self.default_log())

    def test_unicode_brz_log(self):
        uni_val = '§'
        enc = osutils.get_user_encoding()
        try:
            str_val = uni_val.encode(enc)
        except UnicodeEncodeError:
            self.skipTest('Test string {!r} unrepresentable in user encoding {}'.format(uni_val, enc))
        brz_log = os.path.join(self.test_base_dir, uni_val)
        self.overrideEnv('BRZ_LOG', brz_log)
        out, err = self.run_brz_subprocess('version')
        uni_out = out.decode(enc)
        self.assertContainsRe(uni_out, '(?m)^  Breezy log file: .*/§$')