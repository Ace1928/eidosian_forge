from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetServerVolume(command.Command):
    """Update a volume attachment on the server."""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', help=_('Server to update volume for (name or ID)'))
        parser.add_argument('volume', help=_('Volume to update attachment for (name or ID)'))
        termination_group = parser.add_mutually_exclusive_group()
        termination_group.add_argument('--delete-on-termination', action='store_true', dest='delete_on_termination', default=None, help=_('Delete the volume when the server is destroyed (supported by --os-compute-api-version 2.85 or above)'))
        termination_group.add_argument('--preserve-on-termination', action='store_false', dest='delete_on_termination', help=_('Preserve the volume when the server is destroyed (supported by --os-compute-api-version 2.85 or above)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        volume_client = self.app.client_manager.sdk_connection.volume
        if parsed_args.delete_on_termination is not None:
            if not sdk_utils.supports_microversion(compute_client, '2.85'):
                msg = _('--os-compute-api-version 2.85 or greater is required to support the -delete-on-termination or --preserve-on-termination option')
                raise exceptions.CommandError(msg)
            server = compute_client.find_server(parsed_args.server, ignore_missing=False)
            volume = volume_client.find_volume(parsed_args.volume, ignore_missing=False)
            compute_client.update_volume_attachment(server, volume, delete_on_termination=parsed_args.delete_on_termination)