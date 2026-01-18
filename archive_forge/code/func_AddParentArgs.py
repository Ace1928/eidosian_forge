from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddParentArgs(parser, project_help_text, org_help_text, folder_help_text):
    parent_group = parser.add_mutually_exclusive_group(required=True)
    common_args.ProjectArgument(help_text_to_prepend=project_help_text).AddToParser(parent_group)
    AddOrganizationArgs(parent_group, org_help_text)
    AddFolderArgs(parent_group, folder_help_text)