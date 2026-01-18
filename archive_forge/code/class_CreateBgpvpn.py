import logging
from osc_lib.cli import format_columns
from osc_lib.cli.parseractions import KeyValueAction
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
class CreateBgpvpn(command.ShowOne):
    _description = _('Create BGP VPN resource')

    def get_parser(self, prog_name):
        parser = super(CreateBgpvpn, self).get_parser(prog_name)
        nc_osc_utils.add_project_owner_option_to_parser(parser)
        _get_common_parser(parser)
        parser.add_argument('--type', default='l3', choices=['l2', 'l3'], help=_('BGP VPN type selection between IP VPN (l3) and Ethernet VPN (l2) (default: l3)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = {}
        if parsed_args.name is not None:
            attrs['name'] = str(parsed_args.name)
        if parsed_args.type is not None:
            attrs['type'] = parsed_args.type
        if parsed_args.route_targets is not None:
            attrs['route_targets'] = parsed_args.route_targets
        if parsed_args.import_targets is not None:
            attrs['import_targets'] = parsed_args.import_targets
        if parsed_args.export_targets is not None:
            attrs['export_targets'] = parsed_args.export_targets
        if parsed_args.route_distinguishers is not None:
            attrs['route_distinguishers'] = parsed_args.route_distinguishers
        if parsed_args.vni is not None:
            attrs['vni'] = parsed_args.vni
        if parsed_args.local_pref is not None:
            attrs['local_pref'] = parsed_args.local_pref
        if 'project' in parsed_args and parsed_args.project is not None:
            project_id = nc_osc_utils.find_project(self.app.client_manager.identity, parsed_args.project, parsed_args.project_domain).id
            attrs['tenant_id'] = project_id
        obj = client.create_bgpvpn(**attrs)
        display_columns, columns = nc_osc_utils._get_columns(obj)
        data = osc_utils.get_dict_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)