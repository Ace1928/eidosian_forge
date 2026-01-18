from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddUpdateMaskArgToParser(parser):
    """Adds argument for the update-mask flag to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('--update-mask', metavar='UPDATE_MASK', help='Field mask used to specify the fields to be overwritten in the policy by the set. The fields specified in the update_mask are relative to the policy, not the full request. The update-mask flag can be empty, or have values `policy.spec`, `policy.dry_run_spec` or `*`. If the policy does not contain the dry_run_spec and update-mask flag is not provided, then it defaults to `policy.spec`.')