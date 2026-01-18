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
def AddEnableAutoRepairFlag(parser, for_node_pool=False, for_create=False):
    """Adds a --enable-autorepair flag to parser."""
    if for_node_pool:
        help_text = 'Enable node autorepair feature for a node pool.\n\n  $ {command} node-pool-1 --cluster=example-cluster --enable-autorepair\n'
        if for_create:
            help_text += '\nNode autorepair is enabled by default for node pools using COS, COS_CONTAINERD, UBUNTU or UBUNTU_CONTAINERD\nas a base image, use --no-enable-autorepair to disable.\n'
    else:
        help_text = "Enable node autorepair feature for a cluster's default node pool(s).\n\n  $ {command} example-cluster --enable-autorepair\n"
        if for_create:
            help_text += '\nNode autorepair is enabled by default for clusters using COS, COS_CONTAINERD, UBUNTU or UBUNTU_CONTAINERD\nas a base image, use --no-enable-autorepair to disable.\n'
    help_text += '\nSee https://cloud.google.com/kubernetes-engine/docs/how-to/node-auto-repair for more info.'
    parser.add_argument('--enable-autorepair', action='store_true', default=None, help=help_text)