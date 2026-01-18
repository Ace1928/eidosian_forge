from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.resource_manager import tags as rm_tags
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.network_firewall_policies import convert_terraform
from googlecloudsdk.command_lib.compute.network_firewall_policies import secure_tags_utils
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.command_lib.resource_manager import operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _ConvertRuleInternal(messages, firewall, action, l4_configs, tag_mapping):
    return messages.FirewallPolicyRule(disabled=firewall.disabled, ruleName=firewall.name, description=firewall.description, direction=_ConvertRuleDirection(messages, firewall.direction), priority=firewall.priority, action=action, enableLogging=firewall.logConfig.enable, match=messages.FirewallPolicyRuleMatcher(destIpRanges=firewall.destinationRanges, srcIpRanges=firewall.sourceRanges, srcSecureTags=_ConvertTags(messages, tag_mapping, firewall.sourceTags) + _ConvertServiceAccounts(messages, tag_mapping, firewall.sourceServiceAccounts), layer4Configs=_ConvertLayer4Configs(messages, l4_configs)), targetSecureTags=_ConvertTags(messages, tag_mapping, firewall.targetTags) + _ConvertServiceAccounts(messages, tag_mapping, firewall.targetServiceAccounts))