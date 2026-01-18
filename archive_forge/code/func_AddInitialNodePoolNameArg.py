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
def AddInitialNodePoolNameArg(parser, hidden=True):
    """Adds --node-pool-name argument to the parser."""
    help_text = 'Name of the initial node pool that will be created for the cluster.\n\nSpecifies the name to use for the initial node pool that will be created\nwith the cluster.  If the settings specified require multiple node pools\nto be created, the name for each pool will be prefixed by this name.  For\nexample running the following will result in three node pools being\ncreated, example-node-pool-0, example-node-pool-1 and\nexample-node-pool-2:\n\n  $ {command} example-cluster --num-nodes 9 --max-nodes-per-pool 3     --node-pool-name example-node-pool\n'
    parser.add_argument('--node-pool-name', hidden=hidden, help=help_text)