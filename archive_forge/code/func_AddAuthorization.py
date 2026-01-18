from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddAuthorization(parser: parser_arguments.ArgumentInterceptor):
    """Adds flags to specify applied and managed RBAC policy.

  Args:
    parser: The argparse parser to add the flag to.
  """
    authorization_group = parser.add_group(help='User cluster authorization configurations to bootstrap onto the admin cluster')
    flag_help_text = 'Users that will be granted the cluster-admin role on the cluster, providing full access to the cluster.\n\nTo add multiple users, specify one in each flag. When updating, the update command overwrites the whole grant list. Specify all existing and new users that you want to be cluster administrators.\n\nExamples:\n\n  $ {command}\n      --admin-users alice@example.com\n      --admin-users bob@example.com\n'
    authorization_group.add_argument('--admin-users', help=flag_help_text, action='append')