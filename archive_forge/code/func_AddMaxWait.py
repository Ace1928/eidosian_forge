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
def AddMaxWait(parser, default_max_wait, help_text="Time to synchronously wait for the operation to complete, after which the operation continues asynchronously. Ignored if --no-async isn't specified. See $ gcloud topic datetimes for information on time formats."):
    """Adds --max-wait flag."""
    parser.add_argument('--max-wait', dest='max_wait', required=False, default=default_max_wait, help=help_text, type=arg_parsers.Duration())