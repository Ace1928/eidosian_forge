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
def GetCustomConstraintNameFromArgs(args):
    """Returns the custom constraint name from the user-specified arguments.

  This handles both cases in which the user specifies and does not specify the
  custom constraint prefix.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    if args.custom_constraint.startswith(CUSTOM_CONSTRAINT_PREFIX):
        return args.custom_constraint[len(CUSTOM_CONSTRAINT_PREFIX):]
    return args.custom_constraint