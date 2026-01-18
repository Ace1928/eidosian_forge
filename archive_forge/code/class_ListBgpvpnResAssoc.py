import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn import constants
class ListBgpvpnResAssoc(command.Lister):
    """List BGP VPN resource associations for a given BGP VPN"""

    def get_parser(self, prog_name):
        parser = super(ListBgpvpnResAssoc, self).get_parser(prog_name)
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN listed associations belong to (name or ID)'))
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        parser.add_argument('--property', metavar='<key=value>', help=_('Filter property to apply on returned BGP VPNs (repeat to filter on multiple properties)'), action=parseractions.KeyValueAction)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        bgpvpn = client.find_bgpvpn(parsed_args.bgpvpn)
        params = {}
        if parsed_args.property:
            params.update(parsed_args.property)
        if self._assoc_res_name == constants.NETWORK_ASSOC:
            objs = client.bgpvpn_network_associations(bgpvpn['id'], retrieve_all=True, **params)
        elif self._assoc_res_name == constants.PORT_ASSOCS:
            objs = client.bgpvpn_port_associations(bgpvpn['id'], retrieve_all=True, **params)
        else:
            objs = client.bgpvpn_router_associations(bgpvpn['id'], retrieve_all=True, **params)
        transform = getattr(self, '_transform_resource', None)
        transformed_objs = []
        if callable(transform):
            for obj in objs:
                transformed_objs.append(transform(obj))
        else:
            transformed_objs = list(objs)
        headers, columns = column_util.get_column_definitions(self._attr_map, long_listing=parsed_args.long)
        return (headers, (osc_utils.get_dict_properties(s, columns, formatters=self._formatters) for s in transformed_objs))