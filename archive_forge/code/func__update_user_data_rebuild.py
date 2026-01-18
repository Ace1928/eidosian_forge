import copy
import ipaddress
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import port as neutron_port
from heat.engine.resources.openstack.neutron import subnet
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import server_base
from heat.engine import support
from heat.engine import translation
from heat.rpc import api as rpc_api
def _update_user_data_rebuild(self, after_props):
    user_data = after_props[self.USER_DATA]
    prg = progress.ServerUpdateProgress(self.resource_id, 'rebuild', handler_extra={'args': (user_data,)})
    return prg