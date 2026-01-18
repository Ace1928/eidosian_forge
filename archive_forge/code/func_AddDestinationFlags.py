from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddDestinationFlags(parser, required=True):
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument('--gcs-uri', help='Destination Cloud storage bucket where report and evidence must be uploaded. The Cloud storage bucket provided here must be selected among the buckets entered during the enrollment process.')