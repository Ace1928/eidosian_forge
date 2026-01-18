import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class UpdateQosPolicyMixin(CreateQosPolicyMixin):

    def add_arguments_qos_policy(self, parser):
        qos_policy_args = super(UpdateQosPolicyMixin, self).add_arguments_qos_policy(parser)
        qos_policy_args.add_argument('--no-qos-policy', action='store_true', help=_('Detach QoS policy from the resource.'))
        return qos_policy_args

    def args2body_qos_policy(self, parsed_args, resource):
        super(UpdateQosPolicyMixin, self).args2body_qos_policy(parsed_args, resource)
        if parsed_args.no_qos_policy:
            resource['qos_policy_id'] = None