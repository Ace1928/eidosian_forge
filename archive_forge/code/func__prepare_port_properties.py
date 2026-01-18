from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
def _prepare_port_properties(self, props, prepare_for_update=False):
    if not props.pop(self.NO_FIXED_IPS, False):
        if self.FIXED_IPS in props:
            fixed_ips = props[self.FIXED_IPS]
            if fixed_ips:
                for fixed_ip in fixed_ips:
                    for key, value in list(fixed_ip.items()):
                        if value is None:
                            fixed_ip.pop(key)
                    if self.FIXED_IP_SUBNET in fixed_ip:
                        fixed_ip['subnet_id'] = fixed_ip.pop(self.FIXED_IP_SUBNET)
            else:
                del props[self.FIXED_IPS]
    else:
        props[self.FIXED_IPS] = []
    if self.ALLOWED_ADDRESS_PAIRS in props:
        address_pairs = props[self.ALLOWED_ADDRESS_PAIRS]
        if address_pairs:
            for pair in address_pairs:
                if self.ALLOWED_ADDRESS_PAIR_MAC_ADDRESS in pair and pair[self.ALLOWED_ADDRESS_PAIR_MAC_ADDRESS] is None:
                    del pair[self.ALLOWED_ADDRESS_PAIR_MAC_ADDRESS]
        else:
            props[self.ALLOWED_ADDRESS_PAIRS] = []
    if self.SECURITY_GROUPS in props:
        if props.get(self.SECURITY_GROUPS) is not None:
            props[self.SECURITY_GROUPS] = self.client_plugin().get_secgroup_uuids(props.get(self.SECURITY_GROUPS))
        elif prepare_for_update:
            props[self.SECURITY_GROUPS] = self.client_plugin().get_secgroup_uuids(['default'])
    if self.REPLACEMENT_POLICY in props:
        del props[self.REPLACEMENT_POLICY]