from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddOutputFileNameFormatFlag(parser, required=True):
    parser.add_argument('--output-file-name', required=required, help='The name by while scope report should be created .')