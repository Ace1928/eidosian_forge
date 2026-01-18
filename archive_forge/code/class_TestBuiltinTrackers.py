from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
class TestBuiltinTrackers(TestCaseWithMemoryTransport):
    """Test that the builtin trackers are registered and return sane URLs."""

    def test_launchpad_registered(self):
        """The Launchpad bug tracker should be registered by default and
        generate Launchpad bug page URLs.
        """
        branch = self.make_branch('some_branch')
        tracker = bugtracker.tracker_registry.get_tracker('lp', branch)
        self.assertEqual('https://launchpad.net/bugs/1234', tracker.get_bug_url('1234'))

    def test_debian_registered(self):
        """The Debian bug tracker should be registered by default and generate
        bugs.debian.org bug page URLs.
        """
        branch = self.make_branch('some_branch')
        tracker = bugtracker.tracker_registry.get_tracker('deb', branch)
        self.assertEqual('http://bugs.debian.org/1234', tracker.get_bug_url('1234'))

    def test_gnome_registered(self):
        branch = self.make_branch('some_branch')
        tracker = bugtracker.tracker_registry.get_tracker('gnome', branch)
        self.assertEqual('http://bugzilla.gnome.org/show_bug.cgi?id=1234', tracker.get_bug_url('1234'))

    def test_trac_registered(self):
        """The Trac bug tracker should be registered by default and generate
        Trac bug page URLs when the appropriate configuration is present.
        """
        branch = self.make_branch('some_branch')
        config = branch.get_config()
        config.set_user_option('trac_foo_url', 'http://bugs.example.com/trac')
        tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
        self.assertEqual('http://bugs.example.com/trac/ticket/1234', tracker.get_bug_url('1234'))

    def test_bugzilla_registered(self):
        """The Bugzilla bug tracker should be registered by default and
        generate Bugzilla bug page URLs when the appropriate configuration is
        present.
        """
        branch = self.make_branch('some_branch')
        config = branch.get_config()
        config.set_user_option('bugzilla_foo_url', 'http://bugs.example.com')
        tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
        self.assertEqual('http://bugs.example.com/show_bug.cgi?id=1234', tracker.get_bug_url('1234'))

    def test_github(self):
        branch = self.make_branch('some_branch')
        tracker = bugtracker.tracker_registry.get_tracker('github', branch)
        self.assertEqual('https://github.com/breezy-team/breezy/issues/1234', tracker.get_bug_url('breezy-team/breezy/1234'))

    def test_generic_registered(self):
        branch = self.make_branch('some_branch')
        config = branch.get_config()
        config.set_user_option('bugtracker_foo_url', 'http://bugs.example.com/{id}/view.html')
        tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
        self.assertEqual('http://bugs.example.com/1234/view.html', tracker.get_bug_url('1234'))

    def test_generic_registered_non_integer(self):
        branch = self.make_branch('some_branch')
        config = branch.get_config()
        config.set_user_option('bugtracker_foo_url', 'http://bugs.example.com/{id}/view.html')
        tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
        self.assertEqual('http://bugs.example.com/ABC-1234/view.html', tracker.get_bug_url('ABC-1234'))

    def test_generic_incorrect_url(self):
        branch = self.make_branch('some_branch')
        config = branch.get_config()
        config.set_user_option('bugtracker_foo_url', 'http://bugs.example.com/view.html')
        tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
        self.assertRaises(bugtracker.InvalidBugTrackerURL, tracker.get_bug_url, '1234')