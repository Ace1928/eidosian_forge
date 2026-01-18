import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
from neutronclient.osc.v2.networking_bgpvpn import constants
class CreateBgpvpnResAssoc(command.ShowOne):
    """Create a BGP VPN resource association"""
    _action = 'create'

    def get_parser(self, prog_name):
        parser = super(CreateBgpvpnResAssoc, self).get_parser(prog_name)
        nc_osc_utils.add_project_owner_option_to_parser(parser)
        parser.add_argument('bgpvpn', metavar='<bgpvpn>', help=_('BGP VPN to apply the %s association (name or ID)') % self._assoc_res_name)
        parser.add_argument('resource', metavar='<%s>' % self._assoc_res_name, help=_('%s to associate the BGP VPN (name or ID)') % self._assoc_res_name.capitalize())
        get_common_parser = getattr(self, '_get_common_parser', None)
        if callable(get_common_parser):
            get_common_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        bgpvpn = client.find_bgpvpn(parsed_args.bgpvpn)
        find_res_method = getattr(client, 'find_%s' % self._assoc_res_name)
        assoc_res = find_res_method(parsed_args.resource)
        body = {'%s_id' % self._assoc_res_name: assoc_res['id']}
        if 'project' in parsed_args and parsed_args.project is not None:
            project_id = nc_osc_utils.find_project(self.app.client_manager.identity, parsed_args.project, parsed_args.project_domain).id
            body['tenant_id'] = project_id
        arg2body = getattr(self, '_args2body', None)
        if callable(arg2body):
            body.update(arg2body(bgpvpn['id'], parsed_args))
        if self._assoc_res_name == constants.NETWORK_ASSOC:
            obj = client.create_bgpvpn_network_association(bgpvpn['id'], **body)
        elif self._assoc_res_name == constants.PORT_ASSOCS:
            obj = client.create_bgpvpn_port_association(bgpvpn['id'], **body)
        else:
            obj = client.create_bgpvpn_router_association(bgpvpn['id'], **body)
        transform = getattr(self, '_transform_resource', None)
        if callable(transform):
            transform(obj)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = osc_utils.get_dict_properties(obj, columns, formatters=self._formatters)
        return (display_columns, data)