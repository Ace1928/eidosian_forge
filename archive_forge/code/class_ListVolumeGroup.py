import argparse
from cinderclient import api_versions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListVolumeGroup(command.Lister):
    """Lists all volume groups.

    This command requires ``--os-volume-api-version`` 3.13 or greater.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--all-projects', dest='all_projects', action='store_true', default=utils.env('ALL_PROJECTS', default=False), help=_('Shows details for all projects (admin only).'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if volume_client.api_version < api_versions.APIVersion('3.13'):
            msg = _("--os-volume-api-version 3.13 or greater is required to support the 'volume group list' command")
            raise exceptions.CommandError(msg)
        search_opts = {'all_tenants': parsed_args.all_projects}
        groups = volume_client.groups.list(search_opts=search_opts)
        column_headers = ('ID', 'Status', 'Name')
        columns = ('id', 'status', 'name')
        return (column_headers, (utils.get_item_properties(a, columns) for a in groups))