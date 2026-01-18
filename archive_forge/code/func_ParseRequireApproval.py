from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseRequireApproval(trigger, args, messages):
    """Parses approval required flag.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
  """
    if args.require_approval is not None:
        trigger.approvalConfig = messages.ApprovalConfig(approvalRequired=args.require_approval)