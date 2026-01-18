from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerResourceSelectorGroup(parser):
    resource_selector_group = parser.add_group(mutex=False, required=False, help='Specifies a resource for analysis. Leaving it empty means ANY.')
    AddAnalyzerFullResourceNameArgs(resource_selector_group)