import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteVolumeGroup(command.Command):
    """Delete a volume group.

    This command requires ``--os-volume-api-version`` 3.13 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Name or ID of volume group to delete'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Delete the volume group even if it contains volumes. This will delete any remaining volumes in the group.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.13'):
            msg = _("--os-volume-api-version 3.13 or greater is required to support the 'volume group delete' command")
            raise exceptions.CommandError(msg)
        group = utils.find_resource(volume_client.groups, parsed_args.group)
        volume_client.groups.delete(group.id, delete_volumes=parsed_args.force)