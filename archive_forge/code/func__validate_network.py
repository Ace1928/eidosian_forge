import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def _validate_network(self, network):
    net_id = network.get(self.NETWORK_ID)
    port = network.get(self.NETWORK_PORT)
    subnet = network.get(self.NETWORK_SUBNET)
    fixed_ip = network.get(self.NETWORK_FIXED_IP)
    floating_ip = network.get(self.NETWORK_FLOATING_IP)
    str_network = network.get(self.ALLOCATE_NETWORK)
    if net_id is None and port is None and (subnet is None) and (not str_network):
        msg = _('One of the properties "%(id)s", "%(port_id)s", "%(str_network)s" or "%(subnet)s" should be set for the specified network of server "%(server)s".') % dict(id=self.NETWORK_ID, port_id=self.NETWORK_PORT, subnet=self.NETWORK_SUBNET, str_network=self.ALLOCATE_NETWORK, server=self.name)
        raise exception.StackValidationFailed(message=msg)
    has_value_keys = [k for k, v in network.items() if v is not None]
    if str_network and len(has_value_keys) != 1:
        msg = _('Can not specify "%s" with other keys of networks at the same time.') % self.ALLOCATE_NETWORK
        raise exception.StackValidationFailed(message=msg)
    if fixed_ip and port is not None:
        raise exception.ResourcePropertyConflict('/'.join([self.NETWORKS, self.NETWORK_FIXED_IP]), '/'.join([self.NETWORKS, self.NETWORK_PORT]))
    if floating_ip is not None:
        if net_id is not None and port is None and (subnet is None):
            msg = _('Property "%(fip)s" is not supported if only "%(net)s" is specified, because the corresponding port can not be retrieved.') % dict(fip=self.NETWORK_FLOATING_IP, net=self.NETWORK_ID)
            raise exception.StackValidationFailed(message=msg)