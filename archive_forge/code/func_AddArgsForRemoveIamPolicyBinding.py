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
def AddArgsForRemoveIamPolicyBinding(parser, role_completer=None, add_condition=False, condition_completer=None, hide_special_member_types=False):
    """Adds the IAM policy binding arguments for role and members.

  Args:
    parser: An argparse.ArgumentParser-like object to which we add the args.
    role_completer: A command_lib.iam.completers.IamRolesCompleter class to
      complete the --role flag value.
    add_condition: boolean, If true, add the flags for condition.
    condition_completer: A completer to complete the condition flag value.
    hide_special_member_types: boolean. If true, help text for member does not
      include special values `allUsers` and `allAuthenticatedUsers`.

  Raises:
    ArgumentError if one of the arguments is already defined in the parser.
  """
    parser.add_argument('--role', required=True, completer=role_completer, help='The role to remove the principal from.')
    AddMemberFlag(parser, 'to remove the binding for', hide_special_member_types)
    if add_condition:
        _AddConditionFlagsForRemoveBindingFromIamPolicy(parser, condition_completer=condition_completer)