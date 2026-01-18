import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class FirewallPolicyRemoveRule(neutronv20.UpdateCommand):
    """Remove a rule from a given firewall policy."""
    resource = 'firewall_policy'

    def call_api(self, neutron_client, firewall_policy_id, body):
        return neutron_client.firewall_policy_remove_rule(firewall_policy_id, body)

    def args2body(self, parsed_args):
        _rule = ''
        if parsed_args.firewall_rule_id:
            _rule = neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'firewall_rule', parsed_args.firewall_rule_id)
        body = {'firewall_rule_id': _rule}
        return body

    def get_parser(self, prog_name):
        parser = super(FirewallPolicyRemoveRule, self).get_parser(prog_name)
        parser.add_argument('firewall_rule_id', metavar='FIREWALL_RULE', help=_('ID or name of the firewall rule to be removed from the policy.'))
        self.add_known_arguments(parser)
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        body = self.args2body(parsed_args)
        _id = neutronv20.find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.id)
        self.call_api(neutron_client, _id, body)
        print(_('Removed firewall rule from firewall policy %(id)s') % {'id': parsed_args.id}, file=self.app.stdout)