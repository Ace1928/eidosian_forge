import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _upgrade_policies(policies, default_policies):
    old_policies_keys = list(policies.keys())
    for section in sorted(default_policies.keys()):
        rule_defaults = default_policies[section]
        for rule_default in rule_defaults:
            if rule_default.deprecated_rule and rule_default.deprecated_rule.name in old_policies_keys:
                policies[rule_default.name] = policies.pop(rule_default.deprecated_rule.name)
                LOG.info('The name of policy %(old_name)s has been upgraded to%(new_name)', {'old_name': rule_default.deprecated_rule.name, 'new_name': rule_default.name})