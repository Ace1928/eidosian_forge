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
def AddEnableConfidentialStorageFlag(parser, for_node_pool=False, hidden=False):
    """Adds a --enable-confidential-storage flag to the given parser."""
    target = 'node pool' if for_node_pool else 'cluster'
    help_text = 'Enable confidential storage for the {}. Enabling Confidential Storage will\ncreate boot disk with confidential mode\n'.format(target)
    parser.add_argument('--enable-confidential-storage', help=help_text, default=None, hidden=hidden, action='store_true')