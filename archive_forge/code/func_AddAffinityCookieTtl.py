from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddAffinityCookieTtl(parser, hidden=False):
    """Adds affinity cookie Ttl flag to the argparse."""
    affinity_cookie_ttl_help = '      If session-affinity is set to "generated_cookie", this flag sets\n      the TTL, in seconds, of the resulting cookie.  A setting of 0\n      indicates that the cookie should be transient.\n      See $ gcloud topic datetimes for information on duration formats.\n      '
    parser.add_argument('--affinity-cookie-ttl', type=arg_parsers.Duration(), default=None, help=affinity_cookie_ttl_help, hidden=hidden)