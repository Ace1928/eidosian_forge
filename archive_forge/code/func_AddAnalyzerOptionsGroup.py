from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerOptionsGroup(parser, is_sync):
    """Adds a group of options."""
    options_group = GetOrAddOptionGroup(parser)
    AddAnalyzerExpandGroupsArgs(options_group)
    AddAnalyzerExpandRolesArgs(options_group)
    AddAnalyzerExpandResourcesArgs(options_group)
    AddAnalyzerOutputResourceEdgesArgs(options_group)
    AddAnalyzerOutputGroupEdgesArgs(options_group)
    AddAnalyzerAnalyzeServiceAccountImpersonationArgs(options_group)
    if is_sync:
        AddAnalyzerExecutionTimeout(options_group)
        AddAnalyzerShowAccessControlEntries(options_group)