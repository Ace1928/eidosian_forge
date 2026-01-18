from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.network_security.firewall_endpoints import activation_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddDisabledArg(parser, help_text=textwrap.dedent('      Disable a firewall endpoint association. To enable a disabled association, use:\n\n       $ {parent_command} update MY-ASSOCIATION --no-disabled\n\n      ')):
    parser.add_argument('--disabled', action='store_true', default=None, help=help_text)