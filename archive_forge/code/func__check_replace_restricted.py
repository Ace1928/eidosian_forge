from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
def _check_replace_restricted(self, res):
    registry = res.stack.env.registry
    restricted_actions = registry.get_rsrc_restricted_actions(res.name)
    existing_res = self.existing_stack[res.name]
    if 'replace' in restricted_actions:
        ex = exception.ResourceActionRestricted(action='replace')
        failure = exception.ResourceFailure(ex, existing_res, existing_res.UPDATE)
        existing_res._add_event(existing_res.UPDATE, existing_res.FAILED, str(ex))
        raise failure