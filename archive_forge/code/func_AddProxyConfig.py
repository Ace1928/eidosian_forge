from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddProxyConfig(parser):
    """Add proxy configuration flags.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    group = parser.add_argument_group('Proxy config')
    group.add_argument('--proxy-resource-group-id', required=True, help='The ARM ID the of the resource group containing proxy keyvault.')
    group.add_argument('--proxy-secret-id', required=True, help='The URL the of the proxy setting secret with its version.')