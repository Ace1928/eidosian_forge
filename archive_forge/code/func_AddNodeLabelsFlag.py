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
def AddNodeLabelsFlag(parser, for_node_pool=False, for_update=False, hidden=False):
    """Adds a --node-labels flag to the given parser."""
    if for_node_pool:
        if for_update:
            help_text = 'Replaces all the user specified Kubernetes labels on all nodes in an existing\nnode pool with the given labels.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --node-labels=label1=value1,label2=value2\n'
        else:
            help_text = 'Applies the given Kubernetes labels on all nodes in the new node pool.\n\nExamples:\n\n  $ {command} node-pool-1 --cluster=example-cluster --node-labels=label1=value1,label2=value2\n'
    else:
        help_text = 'Applies the given Kubernetes labels on all nodes in the new node pool.\n\nExamples:\n\n  $ {command} example-cluster --node-labels=label-a=value1,label-2=value2\n'
    help_text += '\nNew nodes, including ones created by resize or recreate, will have these labels\non the Kubernetes API node object and can be used in nodeSelectors.\nSee [](http://kubernetes.io/docs/user-guide/node-selection/) for examples.\n\nNote that Kubernetes labels, intended to associate cluster components\nand resources with one another and manage resource lifecycles, are different\nfrom Google Kubernetes Engine labels that are used for the purpose of tracking\nbilling and usage information.'
    parser.add_argument('--node-labels', metavar='NODE_LABEL', type=arg_parsers.ArgDict(), help=help_text, hidden=hidden)