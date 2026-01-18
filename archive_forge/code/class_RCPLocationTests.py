from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
class RCPLocationTests(tests.TestCase):

    def test_without_user(self):
        self.assertEqual('git+ssh://example.com/srv/git/bar', rcp_location_to_url('example.com:/srv/git/bar', scheme='git+ssh'))
        self.assertEqual('ssh://example.com/srv/git/bar', rcp_location_to_url('example.com:/srv/git/bar'))

    def test_with_user(self):
        self.assertEqual('git+ssh://foo@example.com/srv/git/bar', rcp_location_to_url('foo@example.com:/srv/git/bar', scheme='git+ssh'))

    def test_invalid(self):
        self.assertRaises(ValueError, rcp_location_to_url, 'http://srv/git/bar')
        self.assertRaises(ValueError, rcp_location_to_url, 'git/bar')
        self.assertRaises(ValueError, rcp_location_to_url, 'c:/git/bar')