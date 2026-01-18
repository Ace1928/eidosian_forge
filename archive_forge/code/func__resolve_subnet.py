from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
def _resolve_subnet(self, gateway):
    external_gw_fixed_ips = gateway[self.EXTERNAL_GATEWAY_FIXED_IPS]
    for fixed_ip in external_gw_fixed_ips:
        for key, value in fixed_ip.copy().items():
            if value is None:
                fixed_ip.pop(key)
        if self.SUBNET in fixed_ip:
            fixed_ip['subnet_id'] = fixed_ip.pop(self.SUBNET)