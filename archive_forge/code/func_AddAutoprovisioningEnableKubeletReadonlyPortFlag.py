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
def AddAutoprovisioningEnableKubeletReadonlyPortFlag(parser, hidden=True):
    """Adds Kubernetes Read Only Port's enablement flag to the parser."""
    parser.add_argument('--autoprovisioning-enable-insecure-kubelet-readonly-port', default=None, action='store_true', hidden=hidden, help=textwrap.dedent("      Enables the Kubelet's insecure read only port for Autoprovisioned\n      Node Pools.\n\n      If not set, the value from nodePoolDefaults.nodeConfigDefaults will be used.\n\n      To disable the readonly port\n      `--no-autoprovisioning-enable-insecure-kubelet-readonly-port`.\n      "))