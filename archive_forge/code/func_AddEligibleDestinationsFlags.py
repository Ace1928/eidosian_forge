from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddEligibleDestinationsFlags(parser, required=True):
    group = parser.add_group(required=required)
    group.add_argument('--eligible-gcs-buckets', metavar='BUCKET URI', type=arg_parsers.ArgList(min_length=1), help='Eligible cloud storage buckets where report and evidence can be uploaded.')