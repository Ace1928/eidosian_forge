import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
from neutronclient.osc.v2 import utils as v2_utils
class UnsetFirewallGroup(command.Command):
    _description = _('Unset firewall group properties')

    def get_parser(self, prog_name):
        parser = super(UnsetFirewallGroup, self).get_parser(prog_name)
        parser.add_argument(const.FWG, metavar='<firewall-group>', help=_('Firewall group to unset (name or ID)'))
        port_group = parser.add_mutually_exclusive_group()
        port_group.add_argument('--port', metavar='<port>', action='append', help=_('Port(s) (name or ID) to apply firewall group.  This option can be repeated'))
        port_group.add_argument('--all-port', action='store_true', help=_('Remove all ports for this firewall group'))
        parser.add_argument('--ingress-firewall-policy', action='store_true', help=_('Ingress firewall policy (name or ID) to delete'))
        parser.add_argument('--egress-firewall-policy', action='store_true', dest='egress_firewall_policy', help=_('Egress firewall policy (name or ID) to delete'))
        shared_group = parser.add_mutually_exclusive_group()
        shared_group.add_argument('--share', action='store_true', help=_('Restrict use of the firewall group to the current project'))
        parser.add_argument('--enable', action='store_true', help=_('Disable firewall group'))
        return parser

    def _get_attrs(self, client, parsed_args):
        attrs = {}
        if parsed_args.ingress_firewall_policy:
            attrs['ingress_firewall_policy_id'] = None
        if parsed_args.egress_firewall_policy:
            attrs['egress_firewall_policy_id'] = None
        if parsed_args.share:
            attrs['shared'] = False
        if parsed_args.enable:
            attrs['admin_state_up'] = False
        if parsed_args.port:
            old = client.find_firewall_group(parsed_args.firewall_group)['ports']
            new = [client.find_port(r)['id'] for r in parsed_args.port]
            attrs['ports'] = sorted(list(set(old) - set(new)))
        if parsed_args.all_port:
            attrs['ports'] = []
        return attrs

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fwg_id = client.find_firewall_group(parsed_args.firewall_group)['id']
        attrs = self._get_attrs(client, parsed_args)
        try:
            client.update_firewall_group(fwg_id, **attrs)
        except Exception as e:
            msg = _("Failed to unset firewall group '%(group)s': %(e)s") % {'group': parsed_args.firewall_group, 'e': e}
            raise exceptions.CommandError(msg)