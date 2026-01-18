import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ListShareGroupTypeAccess(command.Lister):
    """Get access list for share group type."""
    _description = _('Get access list for share group type (Admin only).')

    def get_parser(self, prog_name):
        parser = super(ListShareGroupTypeAccess, self).get_parser(prog_name)
        parser.add_argument('share_group_type', metavar='<share-group-type>', help=_('Filter results by share group type name or ID.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group_type = apiutils.find_resource(share_client.share_group_types, parsed_args.share_group_type)
        if share_group_type._info.get('is_public'):
            raise exceptions.CommandError('Forbidden to get access list for public share group type.')
        data = share_client.share_group_type_access.list(share_group_type)
        columns = ['Project ID']
        values = (oscutils.get_item_properties(s, columns) for s in data)
        return (columns, values)