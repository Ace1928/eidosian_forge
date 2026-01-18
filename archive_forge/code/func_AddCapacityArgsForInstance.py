from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from cloudsdk.google.protobuf import descriptor_pb2
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core.util import files
def AddCapacityArgsForInstance(require_all_autoscaling_args, hide_autoscaling_args, parser):
    """Parse the instance capacity arguments, including manual and autoscaling.

  Args:
    require_all_autoscaling_args: bool. If True, a complete autoscaling config
      is required.
    hide_autoscaling_args: bool. If True, the autoscaling args will be hidden.
    parser: the argparse parser for the command.
  """
    capacity_parser = parser.add_argument_group(mutex=True, required=False)
    Nodes().AddToParser(capacity_parser)
    ProcessingUnits().AddToParser(capacity_parser)
    autoscaling_config_group_parser = capacity_parser.add_argument_group(help='Autoscaling (Preview)', hidden=hide_autoscaling_args)
    AutoscalingHighPriorityCpuTarget(required=require_all_autoscaling_args).AddToParser(autoscaling_config_group_parser)
    AutoscalingStorageTarget(required=require_all_autoscaling_args).AddToParser(autoscaling_config_group_parser)
    autoscaling_limits_group_parser = autoscaling_config_group_parser.add_argument_group(mutex=True, required=require_all_autoscaling_args)
    autoscaling_node_limits_group_parser = autoscaling_limits_group_parser.add_argument_group(help='Autoscaling limits in nodes')
    AutoscalingMinNodes(required=require_all_autoscaling_args).AddToParser(autoscaling_node_limits_group_parser)
    AutoscalingMaxNodes(required=require_all_autoscaling_args).AddToParser(autoscaling_node_limits_group_parser)
    autoscaling_pu_limits_group_parser = autoscaling_limits_group_parser.add_argument_group(help='Autoscaling limits in processing units')
    AutoscalingMinProcessingUnits(required=require_all_autoscaling_args).AddToParser(autoscaling_pu_limits_group_parser)
    AutoscalingMaxProcessingUnits(required=require_all_autoscaling_args).AddToParser(autoscaling_pu_limits_group_parser)