import sys
from oslo_config import cfg
from oslo_policy import opts
from oslo_policy import policy
def _check_rule(context, rule):
    init()
    credentials = context.to_policy_values()
    try:
        return _ROLE_ENFORCER.authorize(rule, credentials, credentials)
    except policy.PolicyNotRegistered:
        return False