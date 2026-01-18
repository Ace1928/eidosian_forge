from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddInstanceMemoryGibArgToParse(parser, required=True):
    """Adds memory size to create Instance."""
    parser.add_argument('--memory-gib', help='The memory size used to create instance in Gib', type=int, required=required)