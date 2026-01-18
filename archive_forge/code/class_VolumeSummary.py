import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class VolumeSummary(command.ShowOne):
    _description = _('Show a summary of all volumes in this deployment.')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.12'):
            msg = _("--os-volume-api-version 3.12 or greater is required to support the 'volume summary' command")
            raise exceptions.CommandError(msg)
        columns = ['total_count', 'total_size']
        column_headers = ['Total Count', 'Total Size']
        if sdk_utils.supports_microversion(volume_client, '3.36'):
            columns.append('metadata')
            column_headers.append('Metadata')
        all_projects = parsed_args.all_projects
        vol_summary = volume_client.summary(all_projects)
        return (column_headers, utils.get_item_properties(vol_summary, columns, formatters={'metadata': format_columns.DictColumn}))