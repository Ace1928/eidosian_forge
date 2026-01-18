from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def AddDisableAutomateDnsZone(parser):
    """Adds disable automate dns zone flag to the argparse."""
    parser.add_argument('--disable-automate-dns-zone', action='store_true', default=None, help="      If specified, then a DNS zone will not be auto-generated for this Private\n      Service Connect forwarding rule. This can only be specified if the\n      forwarding rule's target is a service attachment\n      (`--target-service-attachment=SERVICE_ATTACHMENT`) or Google APIs bundle\n      (`--target-google-apis-bundle=API_BUNDLE`)\n      ")