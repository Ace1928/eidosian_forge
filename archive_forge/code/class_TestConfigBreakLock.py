from breezy import branch, config, controldir, errors, osutils, tests
from breezy.tests.script import run_script
class TestConfigBreakLock(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.config_file_name = './my.conf'
        self.build_tree_contents([(self.config_file_name, b'[DEFAULT]\none=1\n')])
        self.config = config.LockableConfig(file_name=self.config_file_name)
        self.config.lock_write()

    def test_create_pending_lock(self):
        self.addCleanup(self.config.unlock)
        self.assertTrue(self.config._lock.is_held)

    def test_break_lock(self):
        self.run_bzr('break-lock --config %s' % osutils.dirname(self.config_file_name), stdin='y\n')
        self.assertRaises(errors.LockBroken, self.config.unlock)