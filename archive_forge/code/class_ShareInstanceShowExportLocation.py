from osc_lib.command import command
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ShareInstanceShowExportLocation(command.ShowOne):
    """Display the export location for a share instance."""
    _description = _('Show export location for a share instance.')

    def get_parser(self, prog_name):
        parser = super(ShareInstanceShowExportLocation, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('Name or ID of the share instance'))
        parser.add_argument('export_location', metavar='<export_location>', help=_('ID of the share instance export location.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_instance = osc_utils.find_resource(share_client.share_instances, parsed_args.instance)
        share_instance_export_locations = share_client.share_instance_export_locations.get(share_instance.id, parsed_args.export_location)
        data = share_instance_export_locations._info
        data.pop('links', None)
        return self.dict2columns(data)