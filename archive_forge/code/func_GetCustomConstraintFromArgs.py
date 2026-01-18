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
def GetCustomConstraintFromArgs(args):
    """Returns the CustomConstraint from the user-specified arguments.

  A CustomConstraint has the following syntax:
  organizations/{organization_id}/customConstraints/{constraint_name}.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    organization_id = args.organization
    constraint_name = GetCustomConstraintNameFromArgs(args)
    return 'organizations/{}/customConstraints/{}'.format(organization_id, constraint_name)