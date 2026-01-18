from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import client_exception
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.ec2 import internet_gateway
from heat.engine.resources.aws.ec2 import vpc
from heat.engine import support
def _neutron_add_gateway_router(self, float_id, network_id):
    router = vpc.VPC.router_for_vpc(self.neutron(), network_id)
    if router is not None:
        floatingip = self.neutron().show_floatingip(float_id)
        floating_net_id = floatingip['floatingip']['floating_network_id']
        self.neutron().add_gateway_router(router['id'], {'network_id': floating_net_id})