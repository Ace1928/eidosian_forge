from osc_lib.command import command
from osc_lib import utils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ShareSnapshotInstanceExportLocationShow(command.ShowOne):
    """Show export location of the share snapshot instance."""
    _description = _('Show export location of the share snapshot instance.')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotInstanceExportLocationShow, self).get_parser(prog_name)
        parser.add_argument('snapshot_instance', metavar='<snapshot_instance>', help=_('ID of the share snapshot instance.'))
        parser.add_argument('export_location', metavar='<export_location>', help=_('ID of the share snapshot instance export location.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_instance = apiutils.find_resource(share_client.share_snapshot_instances, parsed_args.snapshot_instance)
        share_snapshot_instance_export_location = share_client.share_snapshot_instance_export_locations.get(parsed_args.export_location, snapshot_instance=snapshot_instance)
        share_snapshot_instance_export_location._info.pop('links', None)
        return self.dict2columns(share_snapshot_instance_export_location._info)