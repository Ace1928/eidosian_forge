import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
from neutronclient.osc.v2 import utils as v2_utils
class ShowFirewallGroup(command.ShowOne):
    _description = _('Display firewall group details')

    def get_parser(self, prog_name):
        parser = super(ShowFirewallGroup, self).get_parser(prog_name)
        parser.add_argument(const.FWG, metavar='<firewall-group>', help=_('Firewall group to show (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fwg_id = client.find_firewall_group(parsed_args.firewall_group)['id']
        obj = client.get_firewall_group(fwg_id)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)