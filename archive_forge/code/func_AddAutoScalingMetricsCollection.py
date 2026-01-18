from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAutoScalingMetricsCollection(parser):
    """Adds autoscaling metrics collection flags.

  Args:
    parser: The argparse.parser to add the arguments to.
  """
    group = parser.add_argument_group('Node pool autoscaling metrics collection')
    AddAutoscalingMetricsGranularity(group, required=True)
    AddAutoscalingMetrics(group)