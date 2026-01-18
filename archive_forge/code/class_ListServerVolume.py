from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListServerVolume(command.Lister):
    """List all the volumes attached to a server."""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', help=_('Server to list volume attachments for (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        volumes = compute_client.volume_attachments(server)
        columns = ()
        column_headers = ()
        if not sdk_utils.supports_microversion(compute_client, '2.89'):
            columns += ('id',)
            column_headers += ('ID',)
        columns += ('device', 'server_id', 'volume_id')
        column_headers += ('Device', 'Server ID', 'Volume ID')
        if sdk_utils.supports_microversion(compute_client, '2.70'):
            columns += ('tag',)
            column_headers += ('Tag',)
        if sdk_utils.supports_microversion(compute_client, '2.79'):
            columns += ('delete_on_termination',)
            column_headers += ('Delete On Termination?',)
        if sdk_utils.supports_microversion(compute_client, '2.89'):
            columns += ('attachment_id', 'bdm_id')
            column_headers += ('Attachment ID', 'BlockDeviceMapping UUID')
        return (column_headers, (utils.get_item_properties(s, columns) for s in volumes))