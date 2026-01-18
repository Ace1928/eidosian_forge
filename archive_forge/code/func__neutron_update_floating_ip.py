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
def _neutron_update_floating_ip(self, allocationId, port_id=None, ignore_not_found=False):
    try:
        self.neutron().update_floatingip(allocationId, {'floatingip': {'port_id': port_id}})
    except Exception as e:
        if not (ignore_not_found and self.client_plugin('neutron').is_not_found(e)):
            raise