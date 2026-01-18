from osc_lib.command import command
from osc_lib import utils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.dynamic_routing import constants
class CreateBgpPeer(command.ShowOne):
    _description = _('Create a BGP peer')

    def get_parser(self, prog_name):
        parser = super(CreateBgpPeer, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the BGP peer to create'))
        parser.add_argument('--peer-ip', metavar='<peer-ip-address>', required=True, help=_('Peer IP address'))
        parser.add_argument('--remote-as', required=True, metavar='<peer-remote-as>', help=_('Peer AS number. (Integer in [%(min_val)s, %(max_val)s] is allowed)') % {'min_val': constants.MIN_AS_NUM, 'max_val': constants.MAX_AS_NUM})
        parser.add_argument('--auth-type', metavar='<peer-auth-type>', choices=['none', 'md5'], type=nc_utils.convert_to_lowercase, default='none', help=_('Authentication algorithm. Supported algorithms: none (default), md5'))
        parser.add_argument('--password', metavar='<auth-password>', help=_('Authentication password'))
        nc_osc_utils.add_project_owner_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        obj = client.create_bgp_peer(**attrs)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)