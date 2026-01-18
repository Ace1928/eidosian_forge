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
def _remove_floating_ip_address(self, eip, ignore_not_found=False):
    try:
        self.client_plugin().dissociate_floatingip_address(eip)
    except Exception as e:
        addr_not_found = isinstance(e, client_exception.EntityMatchNotFound)
        fip_not_found = self.client_plugin().is_not_found(e)
        not_found = addr_not_found or fip_not_found
        if not (ignore_not_found and not_found):
            raise