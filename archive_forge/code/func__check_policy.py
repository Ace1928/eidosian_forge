import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
def _check_policy(self, property_exp, action, context):
    try:
        action = ':'.join([property_exp, action])
        self.policy_enforcer.enforce(context, action, {}, registered=False)
    except exception.Forbidden:
        return False
    return True