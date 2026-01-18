from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.routers.nats.rules import flags
from googlecloudsdk.core import exceptions as core_exceptions
import six
def FindRuleOrRaise(nat, rule_number):
    """Returns the Rule with the given rule_number in the given NAT."""
    for rule in nat.rules:
        if rule.ruleNumber == rule_number:
            return rule
    raise RuleNotFoundError(rule_number)