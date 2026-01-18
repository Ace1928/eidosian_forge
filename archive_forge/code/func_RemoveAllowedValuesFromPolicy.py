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
def RemoveAllowedValuesFromPolicy(policy, args, release_track):
    """Removes the specified allowed values from all policy rules containing the specified condition.

  It searches for and removes the specified values from the
  lists of allowed values on those rules. Any modified rule with empty lists
  of allowed values and denied values after this operation is deleted.

  Args:
    policy: messages.GoogleCloudOrgpolicy{api_version}Policy, The policy to be
      updated.
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
    release_track: calliope.base.ReleaseTrack, Release track of the command.

  Returns:
    The updated policy.
  """
    new_policy = copy.deepcopy(policy)
    if not new_policy.spec.rules:
        return policy
    specified_values = set(args.value)
    for rule_to_update in new_policy.spec.rules:
        if rule_to_update.values is not None:
            rule_to_update.values.allowedValues = [value for value in rule_to_update.values.allowedValues if value not in specified_values]
    return _DeleteRulesWithEmptyValues(new_policy, release_track)