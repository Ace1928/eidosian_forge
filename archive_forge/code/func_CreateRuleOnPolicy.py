from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def CreateRuleOnPolicy(policy, release_track, condition_expression=None):
    """Creates a rule on the policy that contains the specified condition expression.

  In the case that condition_expression is None, a rule without a condition is
  created.

  Args:
    policy: messages.GoogleCloudOrgpolicy{api_version}Policy, The policy object
      to be updated.
    release_track: release track of the command
    condition_expression: str, The condition expression to create a new rule
      with.

  Returns:
    The rule that was created as well as the new policy that includes this
    rule.
  """
    org_policy_api = org_policy_service.OrgPolicyApi(release_track)
    new_policy = copy.deepcopy(policy)
    condition = None
    if condition_expression is not None:
        condition = org_policy_api.messages.GoogleTypeExpr(expression=condition_expression)
    new_rule = org_policy_api.BuildPolicySpecPolicyRule(condition=condition)
    new_policy.spec.rules.append(new_rule)
    return (new_rule, new_policy)