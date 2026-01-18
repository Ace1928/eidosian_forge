from osc_lib.command import command
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ShareReplicaShowExportLocation(command.ShowOne):
    """Show details of a share replica's export location."""
    _description = _("Show details of a share replica's export location.")

    def get_parser(self, prog_name):
        parser = super(ShareReplicaShowExportLocation, self).get_parser(prog_name)
        parser.add_argument('replica', metavar='<replica>', help=_('ID of the share replica.'))
        parser.add_argument('export_location', metavar='<export-location>', help=_('ID of the share replica export location.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        replica = osc_utils.find_resource(share_client.share_replicas, parsed_args.replica)
        export_location = share_client.share_replica_export_locations.get(replica, parsed_args.export_location)
        return self.dict2columns(export_location._info)