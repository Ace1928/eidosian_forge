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
def AddDataplaneV2ObservabilityFlags(parser):
    """Adds --dataplane-v2-observability-mode enum flag and --enable-dataplane-v2-flow-observability, --disable-dataplane-v2-flow-observability boolean flags to parser."""
    group = parser.add_group(mutex=True)
    group.add_argument('--enable-dataplane-v2-flow-observability', action='store_const', const=True, help='Enables Advanced Datapath Observability which allows for a real-time view into pod-to-pod traffic within your cluster.')
    group.add_argument('--disable-dataplane-v2-flow-observability', action='store_const', const=True, help='Disables Advanced Datapath Observability.')
    help_text = '\nSelect Advanced Datapath Observability mode for the cluster. Defaults to `DISABLED`.\n\nAdvanced Datapath Observability allows for a real-time view into pod-to-pod traffic within your cluster.\n\nExamples:\n\n  $ {command} --dataplane-v2-observability-mode=DISABLED\n\n  $ {command} --dataplane-v2-observability-mode=INTERNAL_VPC_LB\n\n  $ {command} --dataplane-v2-observability-mode=EXTERNAL_LB\n'
    group.add_argument('--dataplane-v2-observability-mode', choices=_DPV2_OBS_MODE, help=help_text, action=actions.DeprecationAction('--dataplane-v2-observability-mode', warn='The --dataplane-v2-observability-mode flag is no longer supported. Please use `--enable-dataplane-v2-flow-observability` or `--disable-dataplane-v2-flow-observability`.', removed=True))