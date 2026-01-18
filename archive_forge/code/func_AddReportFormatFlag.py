from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddReportFormatFlag(parser, required=True):
    parser.add_argument('--report-format', required=required, choices=_AUDIT_REPORT_FORMATS, help='The format in which the audit report should be created.')