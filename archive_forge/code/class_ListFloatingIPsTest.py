import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import floatingips
class ListFloatingIPsTest(tests.TestCase):

    def create_list_command(self, list_value):
        mock_floatingip_manager = mock.Mock()
        mock_floatingip_manager.list.return_value = list_value
        mock_client = mock.Mock()
        mock_client.floatingip = mock_floatingip_manager
        blazar_shell = shell.BlazarShell()
        blazar_shell.client = mock_client
        return (floatingips.ListFloatingIPs(blazar_shell, mock.Mock()), mock_floatingip_manager)

    def test_list_floatingips(self):
        list_value = [{'id': '84c4d37e-1f8b-45ce-897b-16ad7f49b0e9'}, {'id': 'f180cf4c-f886-4dd1-8c36-854d17fbefb5'}]
        list_floatingips, floatingip_manager = self.create_list_command(list_value)
        args = argparse.Namespace(sort_by='id', columns=['id'])
        expected = [['id'], [('84c4d37e-1f8b-45ce-897b-16ad7f49b0e9',), ('f180cf4c-f886-4dd1-8c36-854d17fbefb5',)]]
        ret = list_floatingips.get_data(args)
        self.assertEqual(expected[0], ret[0])
        self.assertEqual(expected[1], [x for x in ret[1]])
        floatingip_manager.list.assert_called_once_with(sort_by='id')