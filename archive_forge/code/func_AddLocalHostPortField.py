from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddLocalHostPortField(parser):
    """Adds a --local-host-port flag to the given parser."""
    help_text = '  `LOCAL_HOST:LOCAL_PORT` on which gcloud should bind and listen for connections\n  that should be tunneled.\n\n  `LOCAL_PORT` may be omitted, in which case it is treated as 0 and an arbitrary\n  unused local port is chosen. The colon also may be omitted in that case.\n\n  If `LOCAL_PORT` is 0, an arbitrary unused local port is chosen.'
    parser.add_argument('--local-host-port', type=arg_parsers.HostPort.Parse, default='localhost:0', help=help_text)