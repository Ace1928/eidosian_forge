from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddOrganizationResourceFlagsToParser(parser):
    """Adds flag for the organization ID to the parser.

  Adds --organization flag to the parser. The flag
  is added as required.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('--organization', metavar='ORGANIZATION_ID', required=True, help='Organization ID.')