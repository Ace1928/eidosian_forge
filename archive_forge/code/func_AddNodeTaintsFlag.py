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
def AddNodeTaintsFlag(parser, for_node_pool=False, for_update=False, hidden=False):
    """Adds a --node-taints flag to the given parser."""
    if for_node_pool:
        if for_update:
            help_text = 'Replaces all the user specified Kubernetes taints on all nodes in an existing\nnode pool, which can be used with tolerations for pod scheduling.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --node-taints=key1=val1:NoSchedule,key2=val2:PreferNoSchedule\n'
        else:
            help_text = 'Applies the given kubernetes taints on all nodes in the new node pool, which can\nbe used with tolerations for pod scheduling.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --node-taints=key1=val1:NoSchedule,key2=val2:PreferNoSchedule\n'
    else:
        help_text = 'Applies the given kubernetes taints on all nodes in default node pool(s) in new\ncluster, which can be used with tolerations for pod scheduling.\n\nExamples:\n\n  $ {command} example-cluster --node-taints=key1=val1:NoSchedule,key2=val2:PreferNoSchedule\n'
    help_text += '\nTo read more about node-taints, see https://cloud.google.com/kubernetes-engine/docs/node-taints.\n'
    parser.add_argument('--node-taints', metavar='NODE_TAINT', type=arg_parsers.ArgDict(), help=help_text, hidden=hidden)