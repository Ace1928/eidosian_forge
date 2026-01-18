import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
class DeleteLeaseTestCase(tests.TestCase):

    def create_delete_command(self):
        mock_lease_manager = mock.Mock()
        mock_client = mock.Mock()
        mock_client.lease = mock_lease_manager
        blazar_shell = shell.BlazarShell()
        blazar_shell.client = mock_client
        return (leases.DeleteLease(blazar_shell, mock.Mock()), mock_lease_manager)

    def test_delete_lease(self):
        delete_lease, lease_manager = self.create_delete_command()
        lease_manager.delete.return_value = None
        args = argparse.Namespace(id=FIRST_LEASE)
        delete_lease.run(args)
        lease_manager.delete.assert_called_once_with(FIRST_LEASE)

    def test_delete_lease_by_name(self):
        delete_lease, lease_manager = self.create_delete_command()
        lease_manager.list.return_value = [{'id': FIRST_LEASE, 'name': 'first-lease'}, {'id': SECOND_LEASE, 'name': 'second-lease'}]
        lease_manager.delete.return_value = None
        args = argparse.Namespace(id='second-lease')
        delete_lease.run(args)
        lease_manager.list.assert_called_once_with()
        lease_manager.delete.assert_called_once_with(SECOND_LEASE)