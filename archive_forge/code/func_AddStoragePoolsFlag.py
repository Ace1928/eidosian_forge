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
def AddStoragePoolsFlag(parser, for_node_pool=False, for_create=True, hidden=False):
    """Adds a --storage-pools flag to the given parser."""
    target = 'node pool' if for_node_pool else 'cluster'
    if for_create:
        help_text = "\nA list of storage pools where the {}'s boot disks will be provisioned.\n\nSTORAGE_POOL must be in the format\nprojects/project/zones/zone/storagePools/storagePool\n".format(target)
    else:
        help_text = "A list of storage pools where the {arg1}'s boot disks will be provisioned. Replaces\nall the current storage pools of an existing {arg2}, with the specified storage\npools.\n\nSTORAGE_POOL must be in the format\nprojects/project/zones/zone/storagePools/storagePool\n".format(arg1=target, arg2=target)
    parser.add_argument('--storage-pools', help=help_text, default=None, type=arg_parsers.ArgList(min_length=1), metavar='STORAGE_POOL', hidden=hidden)