import logging
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteVolumeGroupSnapshot(command.Command):
    """Delete a volume group snapshot.

    This command requires ``--os-volume-api-version`` 3.14 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of volume group snapshot to delete'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.14'):
            msg = _("--os-volume-api-version 3.14 or greater is required to support the 'volume group snapshot delete' command")
            raise exceptions.CommandError(msg)
        group_snapshot = volume_client.find_group_snapshot(parsed_args.snapshot, ignore_missing=False, details=False)
        volume_client.delete_group_snapshot(group_snapshot.id)