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
def GetUpdateMaskFromArgs(args):
    """Returns the update-mask from the user-specified arguments.

  This handles both cases in which the user specifies and does not specify the
  policy prefix for partial update of spec or dry_run_spec fields.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    if args.update_mask is None:
        return args.update_mask
    elif args.update_mask.startswith(UPDATE_MASK_POLICY_PREFIX):
        return args.update_mask
    elif args.update_mask == 'spec' or args.update_mask == 'dry_run_spec' or args.update_mask == 'dryRunSpec':
        return UPDATE_MASK_POLICY_PREFIX + args.update_mask
    return args.update_mask