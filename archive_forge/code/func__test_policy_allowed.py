import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def _test_policy_allowed(self, scope, actions, personas):
    enforcer = policy.Enforcer(scope=scope)
    for persona in personas:
        ctx = self._get_context(persona)
        for action in actions:
            enforcer.enforce(ctx, action, target={'project_id': 'test_tenant_id'}, is_registered_policy=True)