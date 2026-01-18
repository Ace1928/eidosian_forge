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
def RemoveBindingFromIamPolicy(policy, member, role):
    """Given an IAM policy, remove bindings as specified by the args.

  An IAM binding is a pair of role and member. Check if the arguments passed
  define both the role and member attribute, search the policy for a binding
  that contains this role and member, and remove it from the policy.

  Args:
    policy: IAM policy from which we want to remove bindings.
    member: The member to remove from the IAM policy.
    role: The role the member should be removed from.

  Raises:
    IamPolicyBindingNotFound: If specified binding is not found.
  """
    for binding in policy.bindings:
        if binding.role == role and member in binding.members:
            binding.members.remove(member)
            break
    else:
        message = 'Policy binding with the specified principal and role not found!'
        raise IamPolicyBindingNotFound(message)
    policy.bindings[:] = [b for b in policy.bindings if b.members]