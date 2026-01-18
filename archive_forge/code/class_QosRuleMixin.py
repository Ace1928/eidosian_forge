from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class QosRuleMixin(object):

    def add_known_arguments(self, parser):
        add_policy_argument(parser)

    def set_extra_attrs(self, parsed_args):
        self.parent_id = qos_policy.get_qos_policy_id(self.get_client(), parsed_args.policy)

    def args2body(self, parsed_args):
        body = {}
        update_policy_args2body(parsed_args, body)
        return {'qos_rule': body}