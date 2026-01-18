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
def GetPolicyNameFromArgs(args):
    """Returns the policy name from the user-specified arguments.

  A policy name has the following syntax:
  [organizations|folders|projects]/{resource_id}/policies/{constraint_name}.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    resource = GetResourceFromArgs(args)
    constraint_name = GetConstraintNameFromArgs(args)
    return '{}/policies/{}'.format(resource, constraint_name)