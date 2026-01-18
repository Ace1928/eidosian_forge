from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@classmethod
def _CreateExclusion(cls, compute_client, target_rule_set, target_rule_ids=None, request_headers=None, request_cookies=None, request_query_params=None, request_uris=None):
    """Creates Exclusion."""
    new_exclusion = compute_client.messages.SecurityPolicyRulePreconfiguredWafConfigExclusion()
    new_exclusion.targetRuleSet = target_rule_set
    for target_rule_id in target_rule_ids or []:
        new_exclusion.targetRuleIds.append(target_rule_id)
    cls._UpdateExclusion(compute_client, new_exclusion, request_headers, request_cookies, request_query_params, request_uris)
    return new_exclusion