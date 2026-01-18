import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
class UnsetFirewallRule(command.Command):
    _description = _('Unset firewall rule properties')

    def get_parser(self, prog_name):
        parser = super(UnsetFirewallRule, self).get_parser(prog_name)
        parser.add_argument(const.FWR, metavar='<firewall-rule>', help=_('Firewall rule to unset (name or ID)'))
        parser.add_argument('--source-ip-address', action='store_true', help=_('Source IP address or subnet'))
        parser.add_argument('--destination-ip-address', action='store_true', help=_('Destination IP address or subnet'))
        parser.add_argument('--source-port', action='store_true', help=_('Source port number or range(integer in [1, 65535] or range like 123:456)'))
        parser.add_argument('--destination-port', action='store_true', help=_('Destination port number or range(integer in [1, 65535] or range like 123:456)'))
        parser.add_argument('--share', action='store_true', help=_('Restrict use of the firewall rule to the current project'))
        parser.add_argument('--enable-rule', action='store_true', help=_('Disable this rule'))
        parser.add_argument('--source-firewall-group', action='store_true', help=_('Source firewall group (name or ID)'))
        parser.add_argument('--destination-firewall-group', action='store_true', help=_('Destination firewall group (name or ID)'))
        return parser

    def _get_attrs(self, client_manager, parsed_args):
        attrs = {}
        if parsed_args.source_ip_address:
            attrs['source_ip_address'] = None
        if parsed_args.source_port:
            attrs['source_port'] = None
        if parsed_args.destination_ip_address:
            attrs['destination_ip_address'] = None
        if parsed_args.destination_port:
            attrs['destination_port'] = None
        if parsed_args.share:
            attrs['shared'] = False
        if parsed_args.enable_rule:
            attrs['enabled'] = False
        if parsed_args.source_firewall_group:
            attrs['source_firewall_group_id'] = None
        if parsed_args.source_firewall_group:
            attrs['destination_firewall_group_id'] = None
        return attrs

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = self._get_attrs(self.app.client_manager, parsed_args)
        fwr_id = client.find_firewall_rule(parsed_args.firewall_rule)['id']
        try:
            client.update_firewall_rule(fwr_id, **attrs)
        except Exception as e:
            msg = _("Failed to unset firewall rule '%(rule)s': %(e)s") % {'rule': parsed_args.firewall_rule, 'e': e}
            raise exceptions.CommandError(msg)