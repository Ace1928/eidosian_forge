from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddParentResourceFlagsToParser(parser):
    """Add flags for the resource ID and enable custom --project flag to be added by modifying the parser.

  Adds --organization, --folder, and --project flags to the parser. The flags
  are added as a required group with a mutex condition, which ensures that the
  user passes in exactly one of the flags.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    resource_group = parser.add_mutually_exclusive_group(required=True, help='Resource that is associated with the organization policy.')
    resource_group.add_argument('--organization', metavar='ORGANIZATION_ID', completer=completers.OrganizationCompleter, help='Organization ID.')
    resource_group.add_argument('--folder', metavar='FOLDER_ID', help='Folder ID.')
    common_args.ProjectArgument(help_text_to_overwrite='Project ID.').AddToParser(resource_group)