import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
def _add_policy_rules(self, property_exp, action, rule):
    """Add policy rules to the policy enforcer.

        For example, if the file listed as property_protection_file has:
        [prop_a]
        create = glance_creator
        then the corresponding policy rule would be:
        "prop_a:create": "rule:glance_creator"
        where glance_creator is defined in policy.yaml. For example:
        "glance_creator": "role:admin or role:glance_create_user"
        """
    rule = 'rule:%s' % rule
    rule_name = '%s:%s' % (property_exp, action)
    rule_dict = policy.Rules.from_dict({rule_name: rule})
    self.policy_enforcer.add_rules(rule_dict)