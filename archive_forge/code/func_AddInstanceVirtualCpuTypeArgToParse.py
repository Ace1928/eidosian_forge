from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddInstanceVirtualCpuTypeArgToParse(parser):
    """Adds virtual CPU Cores argument for Instance."""
    parser.add_argument('--virtual-cpu-type', choices={'UNSPECIFIED': 'Unspecified', 'DEDICATED': 'Dedicated processors. ', 'UNCAPPED_SHARED': 'Uncapped shared processors', 'CAPPED_SHARED': 'Capped shared processors'}, help='Processor type for the instance', type=arg_utils.ChoiceToEnumName, required=True)