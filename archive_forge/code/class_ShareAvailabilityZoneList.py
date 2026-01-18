from osc_lib.command import command
from osc_lib import utils as oscutils
from manilaclient.common._i18n import _
class ShareAvailabilityZoneList(command.Lister):
    """List all availability zones."""
    _description = _('List all availability zones')

    def get_parser(self, prog_name):
        parser = super(ShareAvailabilityZoneList, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        availability_zones = share_client.availability_zones.list()
        fields = ('Id', 'Name', 'Created At', 'Updated At')
        return (fields, (oscutils.get_item_properties(s, fields) for s in availability_zones))