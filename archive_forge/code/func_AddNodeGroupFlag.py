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
def AddNodeGroupFlag(parser):
    """Adds --node-group flag to the parser."""
    help_text = 'Assign instances of this pool to run on the specified Google Compute Engine\nnode group. This is useful for running workloads on sole tenant nodes.\n\nTo see available sole tenant node-groups, run:\n\n  $ gcloud compute sole-tenancy node-groups list\n\nTo create a sole tenant node group, run:\n\n  $ gcloud compute sole-tenancy node-groups create [GROUP_NAME]     --location [ZONE] --node-template [TEMPLATE_NAME]     --target-size [TARGET_SIZE]\n\nSee https://cloud.google.com/compute/docs/nodes for more\ninformation on sole tenancy and node groups.\n'
    parser.add_argument('--node-group', help=help_text)