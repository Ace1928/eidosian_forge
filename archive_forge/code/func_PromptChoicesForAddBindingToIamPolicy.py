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
def PromptChoicesForAddBindingToIamPolicy(policy):
    """The choices in a prompt for condition when adding binding to policy.

  All conditions in the policy will be returned. Two more choices (i.e.
  `None` and `Specify a new condition`) are appended.
  Args:
    policy: the IAM policy which the binding is added to.

  Returns:
    a list of conditions appearing in policy plus the choices of `None` and
    `Specify a new condition`.
  """
    conditions = _ConditionsInPolicy(policy)
    if conditions and conditions[-1][0] != 'None':
        conditions.append(('None', _NONE_CONDITION))
    conditions.append(('Specify a new condition', _NEW_CONDITION))
    return conditions