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
def _data_update_ports(self, port_id, action, port_type='internal_ports'):
    data = self._data_get_ports(port_type)
    if action == 'add':
        data.append({'id': port_id})
    elif action == 'delete':
        for port in data:
            if port_id == port['id']:
                data.remove(port)
                break
    self.data_set(port_type, jsonutils.dumps(data))