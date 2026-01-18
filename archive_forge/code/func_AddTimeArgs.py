from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddTimeArgs(parser):
    """Adds read time."""
    time_group = parser.add_group(mutex=True, required=False, help='Specifies what time period or point in time to query asset metadata at.')
    AddSnapshotTimeArgs(time_group)
    AddReadTimeWindowArgs(time_group)