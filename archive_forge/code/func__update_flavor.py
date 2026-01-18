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
def _update_flavor(self, after_props):
    flavor = after_props[self.FLAVOR]
    handler_args = checker_args = {'args': (flavor,)}
    prg_resize = progress.ServerUpdateProgress(self.resource_id, 'resize', handler_extra=handler_args, checker_extra=checker_args)
    prg_verify = progress.ServerUpdateProgress(self.resource_id, 'verify_resize')
    return (prg_resize, prg_verify)