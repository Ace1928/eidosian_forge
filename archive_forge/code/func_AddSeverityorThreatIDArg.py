from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profiles.threat_prevention import sp_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSeverityorThreatIDArg(parser, required=True):
    """Adds --severities or --threat-ids flag."""
    severity_threatid_args = parser.add_group(mutex=True, required=required)
    severity_threatid_args.add_argument('--severities', type=arg_parsers.ArgList(), metavar='SEVERITY_LEVEL', help='List of comma-separated severities where each value in the list indicates the severity of the threat.')
    severity_threatid_args.add_argument('--threat-ids', type=arg_parsers.ArgList(), metavar='THREAT-ID', help='List of comma-separated threat identifiers where each identifier in the list is a vendor-specified Signature ID representing a threat type. ')