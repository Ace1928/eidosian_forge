from ... import config, tests
from .. import script
from .. import test_config as _t_config
class TestWithoutConfig(tests.TestCaseWithTransport):

    def test_config_all(self):
        out, err = self.run_bzr(['config'])
        self.assertEqual('', out)
        self.assertEqual('', err)

    def test_remove_unknown_option(self):
        self.run_bzr_error(['The "file" configuration option does not exist'], ['config', '--remove', 'file'])

    def test_all_remove_exclusive(self):
        self.run_bzr_error(['--all and --remove are mutually exclusive.'], ['config', '--remove', '--all'])

    def test_all_set_exclusive(self):
        self.run_bzr_error(['Only one option can be set.'], ['config', '--all', 'hello=world'])

    def test_remove_no_option(self):
        self.run_bzr_error(['--remove expects an option to remove.'], ['config', '--remove'])

    def test_unknown_option(self):
        self.run_bzr_error(['The "file" configuration option does not exist'], ['config', 'file'])

    def test_unexpected_regexp(self):
        self.run_bzr_error(['The "\\*file" configuration option does not exist'], ['config', '*file'])

    def test_wrong_regexp(self):
        self.run_bzr_error(['Invalid pattern\\(s\\) found. "\\*file" nothing to repeat'], ['config', '--all', '*file'])