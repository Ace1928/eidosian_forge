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
def AddQueuedProvisioningFlag(parser, hidden=False):
    """Adds a --enable-queued-provisioning flag to parser."""
    parser.add_argument('--enable-queued-provisioning', default=None, help=textwrap.dedent('        Mark the nodepool as Queued only. This means that all new nodes can\n        be obtained only through queuing via ProvisioningRequest API.\n\n          $ {command} node-pool-1 --cluster=example-cluster --enable-queued-provisioning\n          ... and other required parameters, for more details see:\n          https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest\n        '), hidden=hidden, action='store_true')