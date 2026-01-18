from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddAutoscalingSettingsFlagsToParser(parser):
    """Sets up autoscaling settings flags.

  There are two mutually exclusive options to pass the autoscaling settings:
  through command line arguments or as a yaml file.

  Args:
    parser: arg_parser instance that will have the flags added.
  """
    autoscaling_settings_group = parser.add_mutually_exclusive_group(required=False, hidden=True)
    inlined_autoscaling_settings_group = autoscaling_settings_group.add_group()
    inlined_autoscaling_settings_group.add_argument('--autoscaling-min-cluster-node-count', type=int, help='Minimum number of nodes in the cluster')
    inlined_autoscaling_settings_group.add_argument('--autoscaling-max-cluster-node-count', type=int, help='Maximum number of nodes in the cluster')
    inlined_autoscaling_settings_group.add_argument('--autoscaling-cool-down-period', type=str, help='Cool down period (in minutes) between consecutive cluster expansions/contractions')
    inlined_autoscaling_settings_group.add_argument('--autoscaling-policy', type=arg_parsers.ArgDict(spec={'name': str, 'node-type-id': str, 'scale-out-size': int, 'min-node-count': int, 'max-node-count': int, 'cpu-thresholds-scale-in': int, 'cpu-thresholds-scale-out': int, 'granted-memory-thresholds-scale-in': int, 'granted-memory-thresholds-scale-out': int, 'consumed-memory-thresholds-scale-in': int, 'consumed-memory-thresholds-scale-out': int, 'storage-thresholds-scale-in': int, 'storage-thresholds-scale-out': int}, required_keys=['name', 'node-type-id', 'scale-out-size']), action='append', default=list(), help='Autoscaling policy to be applied to the cluster')
    autoscaling_settings_group.add_argument('--autoscaling-settings-from-file', type=arg_parsers.YAMLFileContents(), help='A YAML file containing the autoscaling settings to be applied to the cluster')