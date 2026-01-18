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
def _auto_allocate_network(self):
    topology = self.client('neutron').get_auto_allocated_topology(self.context.tenant_id)['auto_allocated_topology']
    return topology['id']