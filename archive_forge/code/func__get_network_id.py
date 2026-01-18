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
def _get_network_id(self, net):
    net_id = net.get(self.NETWORK_ID) or None
    subnet = net.get(self.NETWORK_SUBNET) or None
    if not net_id and subnet:
        net_id = self.client_plugin('neutron').network_id_from_subnet_id(subnet)
    return net_id