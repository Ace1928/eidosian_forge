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
def _validate_belonging_subnet_to_net(self, network):
    if network.get(self.NETWORK_PORT) is None:
        net = self._get_network_id(network)
        subnet = network.get(self.NETWORK_SUBNET)
        if subnet is not None and net is not None:
            subnet_net = self.client_plugin('neutron').network_id_from_subnet_id(subnet)
            if subnet_net != net:
                msg = _('Specified subnet %(subnet)s does not belongs to network %(network)s.') % {'subnet': subnet, 'network': net}
                raise exception.StackValidationFailed(message=msg)