from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddDataplaneV2MetricsFlag(parser):
    """Adds --enable-dataplane-v2-metrics and --disable-dataplane-v2-metrics boolean flags to parser."""
    group = parser.add_group(mutex=True)
    group.add_argument('--enable-dataplane-v2-metrics', action='store_const', const=True, help='Exposes advanced datapath flow metrics on node port.')
    group.add_argument('--disable-dataplane-v2-metrics', action='store_const', const=True, help='Stops exposing advanced datapath flow metrics on node port.')