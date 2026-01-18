from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
def _resolve_gateway(self, props):
    gateway = props.get(self.EXTERNAL_GATEWAY)
    if gateway:
        gateway['network_id'] = gateway.pop(self.EXTERNAL_GATEWAY_NETWORK)
        if gateway[self.EXTERNAL_GATEWAY_ENABLE_SNAT] is None:
            del gateway[self.EXTERNAL_GATEWAY_ENABLE_SNAT]
        if gateway[self.EXTERNAL_GATEWAY_FIXED_IPS] is None:
            del gateway[self.EXTERNAL_GATEWAY_FIXED_IPS]
        else:
            self._resolve_subnet(gateway)
    return props