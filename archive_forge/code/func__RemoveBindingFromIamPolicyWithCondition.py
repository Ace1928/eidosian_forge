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
def _RemoveBindingFromIamPolicyWithCondition(policy, member, role, condition):
    """Remove the member/role binding with the condition from policy."""
    for binding in policy.bindings:
        if role == binding.role and _EqualConditions(binding_condition=binding.condition, input_condition=condition) and (member in binding.members):
            binding.members.remove(member)
            break
    else:
        raise IamPolicyBindingNotFound('Policy binding with the specified principal, role, and condition not found!')
    policy.bindings[:] = [b for b in policy.bindings if b.members]