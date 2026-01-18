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
def AddEnableNestedVirtualizationFlag(parser, for_node_pool=False, hidden=True):
    target = 'node pool' if for_node_pool else 'default initial node pool'
    help_text = '\n      Enables the use of nested virtualization on the {}.\n      Defaults to `false`. Can only be enabled on UBUNTU_CONTAINERD base image.\n    '.format(target)
    parser.add_argument('--enable-nested-virtualization', help=help_text, default=None, hidden=hidden, action='store_true')