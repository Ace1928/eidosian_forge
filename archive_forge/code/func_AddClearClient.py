from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddClearClient(parser):
    """Adds the --clear-client flag.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    parser.add_argument('--clear-client', action='store_true', default=None, help='Clear the Azure client. This flag is required when updating to use Azure workload identity federation from Azure client to manage  Azure resources.')