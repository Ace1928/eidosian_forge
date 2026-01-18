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
def AddBindingToIamPolicy(binding_message_type, policy, member, role):
    """Given an IAM policy, add new bindings as specified by args.

  An IAM binding is a pair of role and member. Check if the arguments passed
  define both the role and member attribute, create a binding out of their
  values, and append it to the policy.

  Args:
    binding_message_type: The protorpc.Message of the Binding to create
    policy: IAM policy to which we want to add the bindings.
    member: The member to add to IAM policy.
    role: The role the member should have.

  Returns:
    boolean, whether or not the policy was updated.
  """
    for binding in policy.bindings:
        if binding.role == role:
            if member in binding.members:
                return False
    for binding in policy.bindings:
        if binding.role == role:
            binding.members.append(member)
            return True
    policy.bindings.append(binding_message_type(members=[member], role='{0}'.format(role)))
    return True