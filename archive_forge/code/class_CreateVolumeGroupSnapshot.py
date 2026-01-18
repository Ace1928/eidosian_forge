import logging
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateVolumeGroupSnapshot(command.ShowOne):
    """Create a volume group snapshot.

    This command requires ``--os-volume-api-version`` 3.13 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('volume_group', metavar='<volume_group>', help=_('Name or ID of volume group to create a snapshot of.'))
        parser.add_argument('--name', metavar='<name>', help=_('Name of the volume group snapshot.'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of a volume group snapshot.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.14'):
            msg = _("--os-volume-api-version 3.14 or greater is required to support the 'volume group snapshot create' command")
            raise exceptions.CommandError(msg)
        group = volume_client.find_group(parsed_args.volume_group, ignore_missing=False, details=False)
        snapshot = volume_client.create_group_snapshot(group_id=group.id, name=parsed_args.name, description=parsed_args.description)
        return _format_group_snapshot(snapshot)