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
def _AddBindingToIamPolicyWithCondition(binding_message_type, condition_message_type, policy, member, role, condition):
    """Given an IAM policy, add a new role/member binding with condition."""
    for binding in policy.bindings:
        if binding.role == role and _EqualConditions(binding_condition=binding.condition, input_condition=condition):
            if member not in binding.members:
                binding.members.append(member)
            return
    condition_message = None
    if condition is not None:
        condition_message = condition_message_type(expression=condition.get('expression'), title=condition.get('title'), description=condition.get('description'))
    policy.bindings.append(binding_message_type(members=[member], role='{}'.format(role), condition=condition_message))