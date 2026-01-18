from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import stack_user
def _register_access_key(self):

    def access_allowed(resource_name):
        return self._get_user().access_allowed(resource_name)
    self.stack.register_access_allowed_handler(self.resource_id, access_allowed)