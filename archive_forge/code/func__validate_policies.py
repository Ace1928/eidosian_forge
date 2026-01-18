from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import stack_user
def _validate_policies(self, policies):
    for policy in policies or []:
        if not isinstance(policy, str):
            LOG.debug('Ignoring policy %s, must be string resource name', policy)
            continue
        try:
            policy_rsrc = self.stack[policy]
        except KeyError:
            LOG.debug('Policy %(policy)s does not exist in stack %(stack)s', {'policy': policy, 'stack': self.stack.name})
            return False
        if not callable(getattr(policy_rsrc, 'access_allowed', None)):
            LOG.debug('Policy %s is not an AccessPolicy resource', policy)
            return False
    return True