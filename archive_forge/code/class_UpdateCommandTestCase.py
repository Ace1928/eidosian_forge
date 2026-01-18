from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
@testtools.skip('Under construction')
class UpdateCommandTestCase(tests.TestCase):

    def setUp(self):
        super(UpdateCommandTestCase, self).setUp()
        self.app = mock.MagicMock()
        self.update_command = command.UpdateCommand(self.app, [])