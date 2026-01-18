import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class ShowShareSnapshot(command.ShowOne):
    """Display a share snapshot"""
    _description = _('Show details about a share snapshot')

    def get_parser(self, prog_name):
        parser = super(ShowShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the snapshot to display'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_snapshot = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        export_locations = share_client.share_snapshot_export_locations.list(share_snapshot)
        locations = []
        for location in export_locations:
            location._info.pop('links', None)
            locations.append(location._info)
        if parsed_args.formatter == 'table':
            locations = cliutils.convert_dict_list_to_string(locations)
        data = share_snapshot._info
        data['export_locations'] = locations
        data.update({'properties': format_columns.DictColumn(data.pop('metadata', {}))})
        data.pop('links', None)
        return self.dict2columns(data)