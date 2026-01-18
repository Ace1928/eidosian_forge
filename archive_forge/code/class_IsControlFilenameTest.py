from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class IsControlFilenameTest(tests.TestCase):

    def test_is_bzrdir(self):
        self.assertTrue(controldir.is_control_filename('.bzr'))
        self.assertTrue(controldir.is_control_filename('.git'))

    def test_is_not_bzrdir(self):
        self.assertFalse(controldir.is_control_filename('bla'))