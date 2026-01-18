from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils as compute_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetLoggingAggregationIntervalArg(messages):
    return arg_utils.ChoiceEnumMapper('--logging-aggregation-interval', messages.SubnetworkLogConfig.AggregationIntervalValueValuesEnum, custom_mappings={'INTERVAL_5_SEC': 'interval-5-sec', 'INTERVAL_30_SEC': 'interval-30-sec', 'INTERVAL_1_MIN': 'interval-1-min', 'INTERVAL_5_MIN': 'interval-5-min', 'INTERVAL_10_MIN': 'interval-10-min', 'INTERVAL_15_MIN': 'interval-15-min'}, help_str='        Can only be specified if VPC Flow Logs for this subnetwork is\n        enabled. Toggles the aggregation interval for collecting flow logs.\n        Increasing the interval time will reduce the amount of generated flow\n        logs for long lasting connections. Default is an interval of 5 seconds\n        per connection.\n        ')