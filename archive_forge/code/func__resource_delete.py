from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.octavia import octavia_base
from heat.engine import translation
def _resource_delete(self):
    self.client().l7policy_delete(self.resource_id)