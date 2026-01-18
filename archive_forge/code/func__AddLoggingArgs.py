from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddLoggingArgs(parser):
    """Adds arguments to configure NAT logging."""
    enable_logging_help_text = textwrap.dedent('    Enable logging for the NAT. Logs will be exported to Stackdriver. NAT\n    logging is disabled by default.\n    To disable logging for the NAT, use\n    $ {parent_command} update MY-NAT --no-enable-logging --router ROUTER\n      --region REGION')
    log_filter_help_text = textwrap.dedent('    Filter for logs exported to stackdriver.\n\n    The default is ALL.\n\n    If logging is not enabled, filter settings will be persisted but will have\n    no effect.\n\n    Use --[no-]enable-logging to enable and disable logging.\n')
    filter_choices = {'ALL': 'Export logs for all connections handled by this NAT.', 'ERRORS_ONLY': 'Export logs for connection failures only.', 'TRANSLATIONS_ONLY': 'Export logs for successful connections only.'}
    parser.add_argument('--enable-logging', action='store_true', default=None, help=enable_logging_help_text)
    parser.add_argument('--log-filter', help=log_filter_help_text, choices=filter_choices)