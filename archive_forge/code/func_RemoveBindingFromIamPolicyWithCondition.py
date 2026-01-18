from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def RemoveBindingFromIamPolicyWithCondition(policy, member, role, condition, all_conditions=False):
    """Given an IAM policy, remove bindings as specified by the args.

  An IAM binding is a pair of role and member with an optional condition.
  Check if the arguments passed define both the role and member attribute,
  search the policy for a binding that contains this role, member and condition,
  and remove it from the policy.

  Args:
    policy: IAM policy from which we want to remove bindings.
    member: The member to remove from the IAM policy.
    role: The role of the member should be removed from.
    condition: The condition of the binding to be removed.
    all_conditions: If true, all bindings with the specified member and role
      will be removed, regardless of the condition.

  Raises:
    IamPolicyBindingNotFound: If specified binding is not found.
    IamPolicyBindingIncompleteError: when user removes a binding without
      specifying --condition to a policy containing conditions in the
      non-interactive mode.
  """
    if not all_conditions and _PolicyContainsCondition(policy) and (not _ConditionIsSpecified(condition)):
        if not console_io.CanPrompt():
            message = 'Removing a binding without specifying a condition from a policy containing conditions is prohibited in non-interactive mode. Run the command again with `--condition=None` to remove a binding without condition or run command with `--all` to remove all bindings of the specified principal and role.'
            raise IamPolicyBindingIncompleteError(message)
        condition = _PromptForConditionRemoveBindingFromIamPolicy(policy, member, role)
    if all_conditions or _IsAllConditions(condition):
        _RemoveBindingFromIamPolicyAllConditions(policy, member, role)
    else:
        condition = None if _IsNoneCondition(condition) else condition
        _RemoveBindingFromIamPolicyWithCondition(policy, member, role, condition)