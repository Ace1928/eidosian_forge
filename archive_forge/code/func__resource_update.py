from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.octavia import octavia_base
from heat.engine import translation
def _resource_update(self, prop_diff):
    if self.REDIRECT_POOL in prop_diff:
        prop_diff['redirect_pool_id'] = prop_diff.pop(self.REDIRECT_POOL)
    self.client().l7policy_set(self.resource_id, json={'l7policy': prop_diff})