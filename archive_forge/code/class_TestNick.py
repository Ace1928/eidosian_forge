import breezy
from breezy import branch, osutils, tests
class TestNick(tests.TestCaseWithTransport):

    def assertNick(self, expected, working_dir='.', explicit=None, directory=None):
        cmd = ['nick']
        if directory is not None:
            cmd.extend(['--directory', directory])
        actual = self.run_bzr(cmd, working_dir=working_dir)[0][:-1]
        self.assertEqual(expected, actual)
        if explicit is not None:
            br = branch.Branch.open(working_dir)
            conf = br.get_config()
            self.assertEqual(explicit, conf.has_explicit_nickname())
            if explicit:
                self.assertEqual(expected, conf._get_explicit_nickname())

    def test_nick_command(self):
        """brz nick for viewing, setting nicknames"""
        self.make_branch_and_tree('me.dev')
        self.assertNick('me.dev', working_dir='me.dev')
        self.run_bzr('nick moo', working_dir='me.dev')
        self.assertNick('moo', working_dir='me.dev')

    def test_autonick_urlencoded(self):
        self.make_branch_and_tree('!repo')
        self.assertNick('!repo', working_dir='!repo')

    def test_bound_nick(self):
        """Bind should not update implicit nick."""
        base = self.make_branch_and_tree('base')
        child = self.make_branch_and_tree('child')
        self.assertNick('child', working_dir='child', explicit=False)
        self.run_bzr('bind ../base', working_dir='child')
        self.assertNick(base.branch.nick, working_dir='child', explicit=False)

    def test_bound_nick_explicit(self):
        """Bind should update explicit nick."""
        base = self.make_branch_and_tree('base')
        child = self.make_branch_and_tree('child')
        self.run_bzr('nick explicit_nick', working_dir='child')
        self.assertNick('explicit_nick', working_dir='child', explicit=True)
        self.run_bzr('bind ../base', working_dir='child')
        self.assertNick(base.branch.nick, working_dir='child', explicit=True)

    def test_boundless_nick(self):
        """Nick defaults to implicit local nick when bound branch is AWOL"""
        base = self.make_branch_and_tree('base')
        child = self.make_branch_and_tree('child')
        self.run_bzr('bind ../base', working_dir='child')
        self.assertNick(base.branch.nick, working_dir='child', explicit=False)
        osutils.rmtree('base')
        self.assertNick('child', working_dir='child', explicit=False)

    def test_nick_directory(self):
        """Test --directory option"""
        self.make_branch_and_tree('me.dev')
        self.assertNick('me.dev', directory='me.dev')
        self.run_bzr(['nick', '-d', 'me.dev', 'moo'])
        self.assertNick('moo', directory='me.dev')