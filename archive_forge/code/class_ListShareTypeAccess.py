from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ListShareTypeAccess(command.Lister):
    """Get access list for share type."""
    _description = _('Get access list for share type')

    def get_parser(self, prog_name):
        parser = super(ListShareTypeAccess, self).get_parser(prog_name)
        parser.add_argument('share_type', metavar='<share_type>', help=_('Share type name or ID to get access list for'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type)
        if share_type._info.get('share_type_access:is_public'):
            raise exceptions.CommandError('Forbidden to get access list for public share type.')
        data = share_client.share_type_access.list(share_type)
        columns = ['Project ID']
        values = (oscutils.get_item_properties(s, columns) for s in data)
        return (columns, values)