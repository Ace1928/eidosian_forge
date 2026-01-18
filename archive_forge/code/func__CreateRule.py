from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def _CreateRule(rule_yaml, compute_holder):
    """Creates a Rule object from the given parsed YAML."""
    rule = compute_holder.client.messages.RouterNatRule()
    if 'ruleNumber' in rule_yaml:
        rule.ruleNumber = rule_yaml['ruleNumber']
    if 'match' in rule_yaml:
        rule.match = rule_yaml['match']
    if 'action' in rule_yaml:
        action_yaml = rule_yaml['action']
        rule.action = compute_holder.client.messages.RouterNatRuleAction()
        if 'sourceNatActiveIps' in action_yaml:
            rule.action.sourceNatActiveIps = action_yaml['sourceNatActiveIps']
        if 'sourceNatDrainIps' in action_yaml:
            rule.action.sourceNatDrainIps = action_yaml['sourceNatDrainIps']
        if 'sourceNatActiveRanges' in action_yaml:
            rule.action.sourceNatActiveRanges = action_yaml['sourceNatActiveRanges']
        if 'sourceNatDrainRanges' in action_yaml:
            rule.action.sourceNatDrainRanges = action_yaml['sourceNatDrainRanges']
    return rule