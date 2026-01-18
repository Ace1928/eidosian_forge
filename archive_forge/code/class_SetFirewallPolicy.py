import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
class SetFirewallPolicy(command.Command):
    _description = _('Set firewall policy properties')

    def get_parser(self, prog_name):
        parser = super(SetFirewallPolicy, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument(const.FWP, metavar='<firewall-policy>', help=_('Firewall policy to update (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Name for the firewall policy'))
        parser.add_argument('--firewall-rule', action='append', metavar='<firewall-rule>', help=_('Firewall rule(s) to apply (name or ID)'))
        parser.add_argument('--no-firewall-rule', action='store_true', help=_('Remove all firewall rules from firewall policy'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fwp_id = client.find_firewall_policy(parsed_args.firewall_policy)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        try:
            client.update_firewall_policy(fwp_id, **attrs)
        except Exception as e:
            msg = _("Failed to set firewall policy '%(policy)s': %(e)s") % {'policy': parsed_args.firewall_policy, 'e': e}
            raise exceptions.CommandError(msg)