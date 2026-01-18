import io
import re
import sys
import fixtures
import testtools
from blazarclient import shell
from blazarclient import tests
class BlazarShellTestCase(tests.TestCase):

    def make_env(self, exclude=None, fake_env=FAKE_ENV):
        env = dict(((k, v) for k, v in fake_env.items() if k != exclude))
        self.useFixture(fixtures.MonkeyPatch('os.environ', env))

    def setUp(self):
        super(BlazarShellTestCase, self).setUp()
        self.blazar_shell = shell.BlazarShell()

    def shell(self, argstr, exitcodes=(0,)):
        orig = sys.stdout
        orig_stderr = sys.stderr
        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            _shell = shell.BlazarShell()
            _shell.initialize_app(argstr.split())
        except SystemExit:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.assertIn(exc_value.code, exitcodes)
        finally:
            stdout = sys.stdout.getvalue()
            sys.stdout.close()
            sys.stdout = orig
            stderr = sys.stderr.getvalue()
            sys.stderr.close()
            sys.stderr = orig_stderr
        return (stdout, stderr)

    def test_help_unknown_command(self):
        self.assertRaises(ValueError, self.shell, 'bash-completion')

    @testtools.skip('lol')
    def test_bash_completion(self):
        stdout, stderr = self.shell('bash-completion')
        required = ['.*--matching', '.*--wrap', '.*help', '.*secgroup-delete-rule', '.*--priority']
        for r in required:
            self.assertThat(stdout + stderr, testtools.matchers.MatchesRegex(r, re.DOTALL | re.MULTILINE))

    @testtools.skip('lol')
    def test_authenticate_user(self):
        obj = shell.BlazarShell()
        obj.initialize_app('list-leases')
        obj.options.os_token = 'aaaa-bbbb-cccc'
        obj.options.os_cacert = 'cert'
        obj.authenticate_user()