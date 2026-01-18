from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
class CreateCommandTestCase(tests.TestCase):

    def setUp(self):
        super(CreateCommandTestCase, self).setUp()
        self.app = mock.MagicMock()
        self.create_command = command.CreateCommand(self.app, [])
        self.client = self.patch(self.create_command, 'get_client')

    @testtools.skip('Under construction')
    def test_get_data_data(self):
        data = {'key_string': 'string_value', 'key_dict': "{'key0': 'value', 'key1': 'value'}", 'key_list': "['1', '2', '3',]", 'key_none': None}
        self.client.resource.return_value = mock.MagicMock(return_value=data)
        self.assertEqual(self.create_command.get_data({'a': 'b'}), None)