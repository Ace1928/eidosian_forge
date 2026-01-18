from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddProjectOrFolderFlags(parser, help_text, required=True):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--project', help='Project Id {}'.format(help_text))
    group.add_argument('--folder', help='Folder Id {}'.format(help_text))