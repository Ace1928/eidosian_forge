from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddOutputDirectoryFormatFlag(parser, required=False):
    parser.add_argument('--output-directory', required=required, help='The directory path where the scope report should be created .')