from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddMonitoringConfig(parser, for_create=False):
    """Adds --enable-managed-prometheus and --disable-managed-prometheus flags to parser."""
    enable_help_text = '\n  Enables managed collection for Managed Service for Prometheus in the cluster.\n\n  See https://cloud.google.com/stackdriver/docs/managed-prometheus/setup-managed#enable-mgdcoll-gke\n  for more info.\n\n  Enabled by default for cluster versions 1.27 or greater,\n  use --no-enable-managed-prometheus to disable.\n  '
    if for_create:
        parser.add_argument('--enable-managed-prometheus', action='store_true', default=None, help=enable_help_text)
    else:
        group = parser.add_group('Monitoring Config', mutex=True)
        group.add_argument('--disable-managed-prometheus', action='store_true', default=None, help='Disable managed collection for Managed Service for Prometheus.')
        group.add_argument('--enable-managed-prometheus', action='store_true', default=None, help='Enable managed collection for Managed Service for Prometheus.')