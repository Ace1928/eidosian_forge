from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAutoscalingMetricsGranularity(parser, required=False):
    parser.add_argument('--autoscaling-metrics-granularity', required=required, help='Frequency at which EC2 Auto Scaling sends aggregated data to AWS CloudWatch. The only valid value is "1Minute".')