from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
class BlazarCommandTestCase(tests.TestCase):

    def setUp(self):
        super(BlazarCommandTestCase, self).setUp()
        self.app = mock.MagicMock()
        self.parser = self.patch(command.OpenStackCommand, 'get_parser')
        self.command = command.BlazarCommand(self.app, [])

    def test_get_client(self):
        client_manager = self.app.client_manager
        del self.app.client_manager
        client = self.command.get_client()
        self.assertEqual(self.app.client, client)
        self.app.client_manager = client_manager
        del self.app.client
        client = self.command.get_client()
        self.assertEqual(self.app.client_manager.reservation, client)

    def test_get_parser(self):
        self.command.get_parser('TestCase')
        self.parser.assert_called_once_with('TestCase')

    def test_format_output_data(self):
        data_before = {'key_string': 'string_value', 'key_dict': {'key': 'value'}, 'key_list': ['1', '2', '3'], 'key_none': None}
        data_after = {'key_string': 'string_value', 'key_dict': '{"key": "value"}', 'key_list': '1\n2\n3', 'key_none': ''}
        self.command.format_output_data(data_before)
        self.assertEqual(data_after, data_before)