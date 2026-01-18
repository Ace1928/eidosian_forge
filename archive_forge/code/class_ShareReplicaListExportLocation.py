from osc_lib.command import command
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ShareReplicaListExportLocation(command.Lister):
    """List export locations of a share replica."""
    _description = _('List export locations of a share replica.')

    def get_parser(self, prog_name):
        parser = super(ShareReplicaListExportLocation, self).get_parser(prog_name)
        parser.add_argument('replica', metavar='<replica>', help=_('ID of the share replica.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        columns = ['ID', 'Availability Zone', 'Replica State', 'Preferred', 'Path']
        replica = osc_utils.find_resource(share_client.share_replicas, parsed_args.replica)
        export_locations = share_client.share_replica_export_locations.list(replica)
        data = (osc_utils.get_dict_properties(location._info, columns) for location in export_locations)
        return (columns, data)