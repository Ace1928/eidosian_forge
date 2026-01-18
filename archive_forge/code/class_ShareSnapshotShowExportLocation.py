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
class ShareSnapshotShowExportLocation(command.ShowOne):
    """Show export location of the share snapshot"""
    _description = _('Show export location of the share snapshot')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotShowExportLocation, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the share snapshot.'))
        parser.add_argument('export_location', metavar='<export-location>', help=_('ID of the share snapshot export location.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_obj = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        export_location = share_client.share_snapshot_export_locations.get(export_location=parsed_args.export_location, snapshot=snapshot_obj)
        return self.dict2columns(export_location._info)