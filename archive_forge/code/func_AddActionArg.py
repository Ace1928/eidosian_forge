from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profiles.threat_prevention import sp_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddActionArg(parser, actions=None, required=True):
    choices = actions or DEFAULT_ACTIONS
    parser.add_argument('--action', required=required, choices=choices, help='Action associated with severity or threat-id')