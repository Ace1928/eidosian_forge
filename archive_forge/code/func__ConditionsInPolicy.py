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
def _ConditionsInPolicy(policy, member=None, role=None):
    """Select conditions in bindings which have the given role and member.

  Search bindings from policy and return their conditions which has the given
  role and member if role and member are given. If member and role are not
  given, return all conditions. Duplicates are not returned.

  Args:
    policy: IAM policy to collect conditions
    member: member which should appear in the binding to select its condition
    role: role which should be the role of binding to select its condition

  Returns:
    A list of conditions got selected
  """
    conditions = {}
    for binding in policy.bindings:
        if (member is None or member in binding.members) and (role is None or role == binding.role):
            condition = binding.condition
            conditions[_ConditionToString(condition)] = condition
    contain_none = False
    if 'None' in conditions:
        contain_none = True
        del conditions['None']
    conditions = [(condition_str, condition) for condition_str, condition in conditions.items()]
    conditions = sorted(conditions, key=lambda x: x[0])
    if contain_none:
        conditions.append(('None', _NONE_CONDITION))
    return conditions