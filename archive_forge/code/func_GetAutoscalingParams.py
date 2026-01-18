from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetAutoscalingParams(args):
    min_nodes = 0
    max_nodes = 0
    min_nodes = args.min_nodes
    max_nodes = args.max_nodes
    return (min_nodes, max_nodes)