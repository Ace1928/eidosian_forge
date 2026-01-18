from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
@testtools.skip('Under construction')
class ShowCommandTestCase(tests.TestCase):

    def setUp(self):
        super(ShowCommandTestCase, self).setUp()
        self.app = mock.MagicMock()
        self.show_command = command.ShowCommand(self.app, [])