from twisted.python import usage
from twisted.trial import unittest
class ZshCompleterTests(unittest.TestCase):
    """
    Test the behavior of the various L{twisted.usage.Completer} classes
    for producing output usable by zsh tab-completion system.
    """

    def test_completer(self):
        """
        Completer produces zsh shell-code that produces no completion matches.
        """
        c = usage.Completer()
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ':some-option:')
        c = usage.Completer(descr='some action', repeat=True)
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, '*:some action:')

    def test_files(self):
        """
        CompleteFiles produces zsh shell-code that completes file names
        according to a glob.
        """
        c = usage.CompleteFiles()
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ':some-option (*):_files -g "*"')
        c = usage.CompleteFiles('*.py')
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ':some-option (*.py):_files -g "*.py"')
        c = usage.CompleteFiles('*.py', descr='some action', repeat=True)
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, '*:some action (*.py):_files -g "*.py"')

    def test_dirs(self):
        """
        CompleteDirs produces zsh shell-code that completes directory names.
        """
        c = usage.CompleteDirs()
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ':some-option:_directories')
        c = usage.CompleteDirs(descr='some action', repeat=True)
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, '*:some action:_directories')

    def test_list(self):
        """
        CompleteList produces zsh shell-code that completes words from a fixed
        list of possibilities.
        """
        c = usage.CompleteList('ABC')
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ':some-option:(A B C)')
        c = usage.CompleteList(['1', '2', '3'])
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ':some-option:(1 2 3)')
        c = usage.CompleteList(['1', '2', '3'], descr='some action', repeat=True)
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, '*:some action:(1 2 3)')

    def test_multiList(self):
        """
        CompleteMultiList produces zsh shell-code that completes multiple
        comma-separated words from a fixed list of possibilities.
        """
        c = usage.CompleteMultiList('ABC')
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ":some-option:_values -s , 'some-option' A B C")
        c = usage.CompleteMultiList(['1', '2', '3'])
        got = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(got, ":some-option:_values -s , 'some-option' 1 2 3")
        c = usage.CompleteMultiList(['1', '2', '3'], descr='some action', repeat=True)
        got = c._shellCode('some-option', usage._ZSH)
        expected = "*:some action:_values -s , 'some action' 1 2 3"
        self.assertEqual(got, expected)

    def test_usernames(self):
        """
        CompleteUsernames produces zsh shell-code that completes system
        usernames.
        """
        c = usage.CompleteUsernames()
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, ':some-option:_users')
        c = usage.CompleteUsernames(descr='some action', repeat=True)
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, '*:some action:_users')

    def test_groups(self):
        """
        CompleteGroups produces zsh shell-code that completes system group
        names.
        """
        c = usage.CompleteGroups()
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, ':group:_groups')
        c = usage.CompleteGroups(descr='some action', repeat=True)
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, '*:some action:_groups')

    def test_hostnames(self):
        """
        CompleteHostnames produces zsh shell-code that completes hostnames.
        """
        c = usage.CompleteHostnames()
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, ':some-option:_hosts')
        c = usage.CompleteHostnames(descr='some action', repeat=True)
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, '*:some action:_hosts')

    def test_userAtHost(self):
        """
        CompleteUserAtHost produces zsh shell-code that completes hostnames or
        a word of the form <username>@<hostname>.
        """
        c = usage.CompleteUserAtHost()
        out = c._shellCode('some-option', usage._ZSH)
        self.assertTrue(out.startswith(':host | user@host:'))
        c = usage.CompleteUserAtHost(descr='some action', repeat=True)
        out = c._shellCode('some-option', usage._ZSH)
        self.assertTrue(out.startswith('*:some action:'))

    def test_netInterfaces(self):
        """
        CompleteNetInterfaces produces zsh shell-code that completes system
        network interface names.
        """
        c = usage.CompleteNetInterfaces()
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, ':some-option:_net_interfaces')
        c = usage.CompleteNetInterfaces(descr='some action', repeat=True)
        out = c._shellCode('some-option', usage._ZSH)
        self.assertEqual(out, '*:some action:_net_interfaces')