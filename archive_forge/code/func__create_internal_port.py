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
def _create_internal_port(self, net_data, net_number, security_groups=None):
    name = _('%(server)s-port-%(number)s') % {'server': self.name, 'number': net_number}
    kwargs = self._prepare_internal_port_kwargs(net_data, security_groups)
    kwargs['name'] = name
    port = self.client('neutron').create_port({'port': kwargs})['port']
    self._data_update_ports(port['id'], 'add')
    return port['id']