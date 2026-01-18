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
def _get_available_networks(self):
    search_opts = {'tenant_id': self.context.tenant_id, 'shared': False, 'admin_state_up': True}
    nc = self.client('neutron')
    nets = nc.list_networks(**search_opts).get('networks', [])
    search_opts = {'shared': True}
    nets += nc.list_networks(**search_opts).get('networks', [])
    ids = [net['id'] for net in nets]
    return ids