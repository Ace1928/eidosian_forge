import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn import constants
class ShowBgpvpnResAssoc(command.ShowOne):
    """Show information of a given BGP VPN resource association"""

    def get_parser(self, prog_name):
        parser = super(ShowBgpvpnResAssoc, self).get_parser(prog_name)
        parser.add_argument('resource_association_id', metavar='<%s association ID>' % self._assoc_res_name, help=_('%s association ID to look up') % self._assoc_res_name.capitalize())
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN the association belongs to (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        bgpvpn = client.find_bgpvpn(parsed_args.bgpvpn)
        if self._assoc_res_name == constants.NETWORK_ASSOC:
            obj = client.get_bgpvpn_network_association(bgpvpn['id'], parsed_args.resource_association_id)
        elif self._assoc_res_name == constants.PORT_ASSOCS:
            obj = client.get_bgpvpn_port_association(bgpvpn['id'], parsed_args.resource_association_id)
        else:
            obj = client.get_bgpvpn_router_association(bgpvpn['id'], parsed_args.resource_association_id)
        transform = getattr(self, '_transform_resource', None)
        if callable(transform):
            transform(obj)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = osc_utils.get_dict_properties(obj, columns, formatters=self._formatters)
        return (display_columns, data)