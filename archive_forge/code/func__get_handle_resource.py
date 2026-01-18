from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
def _get_handle_resource(self):
    return self.stack.resource_by_refid(self.properties[self.HANDLE])