from breezy import tests
from breezy.tests import features
class TestLSProf(tests.TestCaseInTempDir):
    _test_needs_features = [features.lsprof_feature]

    def test_file(self):
        out, err = self.run_bzr('--lsprof-file output.callgrind rocks')
        self.assertNotContainsRe(out, 'Profile data written to')
        self.assertContainsRe(err, 'Profile data written to')

    def test_stdout(self):
        out, err = self.run_bzr('--lsprof rocks')
        self.assertContainsRe(out, 'CallCount')
        self.assertNotContainsRe(err, 'Profile data written to')