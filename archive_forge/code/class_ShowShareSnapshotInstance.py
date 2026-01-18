from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class ShowShareSnapshotInstance(command.ShowOne):
    """Show details about a share snapshot instance."""
    _description = _('Show details about a share snapshot instance.')

    def get_parser(self, prog_name):
        parser = super(ShowShareSnapshotInstance, self).get_parser(prog_name)
        parser.add_argument('snapshot_instance', metavar='<snapshot_instance>', help=_('ID of the share snapshot instance.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_instance = share_client.share_snapshot_instances.get(parsed_args.snapshot_instance)
        snapshot_instance_export_locations = share_client.share_snapshot_instance_export_locations.list(snapshot_instance=snapshot_instance)
        snapshot_instance._info['export_locations'] = []
        for element_location in snapshot_instance_export_locations:
            element_location._info.pop('links', None)
            snapshot_instance._info['export_locations'].append(element_location._info)
        if parsed_args.formatter == 'table':
            snapshot_instance._info['export_locations'] = cliutils.convert_dict_list_to_string(snapshot_instance._info['export_locations'])
        snapshot_instance._info.pop('links', None)
        return self.dict2columns(snapshot_instance._info)