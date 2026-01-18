from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateAdaptiveProtectionConfig(client, args, existing_adaptive_protection_config):
    """Returns a SecurityPolicyAdaptiveProtectionConfig message."""
    messages = client.messages
    adaptive_protection_config = existing_adaptive_protection_config if existing_adaptive_protection_config is not None else messages.SecurityPolicyAdaptiveProtectionConfig()
    if args.IsSpecified('enable_layer7_ddos_defense') or args.IsSpecified('layer7_ddos_defense_rule_visibility'):
        layer7_ddos_defense_config = adaptive_protection_config.layer7DdosDefenseConfig if adaptive_protection_config.layer7DdosDefenseConfig is not None else messages.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig()
        if args.IsSpecified('enable_layer7_ddos_defense'):
            layer7_ddos_defense_config.enable = args.enable_layer7_ddos_defense
        if args.IsSpecified('layer7_ddos_defense_rule_visibility'):
            layer7_ddos_defense_config.ruleVisibility = messages.SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfig.RuleVisibilityValueValuesEnum(args.layer7_ddos_defense_rule_visibility)
        adaptive_protection_config.layer7DdosDefenseConfig = layer7_ddos_defense_config
    return adaptive_protection_config