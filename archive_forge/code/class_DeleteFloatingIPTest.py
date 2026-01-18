import argparse
from unittest import mock
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import floatingips
class DeleteFloatingIPTest(tests.TestCase):

    def create_delete_command(self, list_value):
        mock_floatingip_manager = mock.Mock()
        mock_floatingip_manager.list.return_value = list_value
        mock_client = mock.Mock()
        mock_client.floatingip = mock_floatingip_manager
        blazar_shell = shell.BlazarShell()
        blazar_shell.client = mock_client
        return (floatingips.DeleteFloatingIP(blazar_shell, mock.Mock()), mock_floatingip_manager)

    def test_delete_floatingip(self):
        list_value = [{'id': '84c4d37e-1f8b-45ce-897b-16ad7f49b0e9'}, {'id': 'f180cf4c-f886-4dd1-8c36-854d17fbefb5'}]
        delete_floatingip, floatingip_manager = self.create_delete_command(list_value)
        args = argparse.Namespace(id='84c4d37e-1f8b-45ce-897b-16ad7f49b0e9')
        delete_floatingip.run(args)
        floatingip_manager.delete.assert_called_once_with('84c4d37e-1f8b-45ce-897b-16ad7f49b0e9')