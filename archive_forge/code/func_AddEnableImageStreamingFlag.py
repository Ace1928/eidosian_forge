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
def AddEnableImageStreamingFlag(parser, for_node_pool=False):
    """Adds the argument to handle image streaming configurations."""
    target = 'node pool' if for_node_pool else 'cluster'
    help_text = 'Specifies whether to enable image streaming on {}.'.format(target)
    parser.add_argument('--enable-image-streaming', help=help_text, default=None, action='store_true')