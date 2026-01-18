from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddFleetProject(parser):
    parser.add_argument('--fleet-project', type=arg_parsers.CustomFunctionValidator(project_util.ValidateProjectIdentifier, '--fleet-project must be a valid project ID or project number.'), required=True, help='ID or number of the Fleet host project where the cluster is registered.')