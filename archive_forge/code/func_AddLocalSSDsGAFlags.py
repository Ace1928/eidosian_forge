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
def AddLocalSSDsGAFlags(parser, for_node_pool=False, suppressed=False):
    """Adds the --local-ssd-count, --local-nvme-ssd-block and --ephemeral-storage-local-ssd flags to the parser."""
    group = parser.add_mutually_exclusive_group()
    AddLocalSSDFlag(group, suppressed=suppressed)
    AddEphemeralStorageLocalSSDFlag(group, for_node_pool=for_node_pool, hidden=suppressed)
    AddLocalNvmeSSDBlockFlag(group, for_node_pool=for_node_pool, hidden=suppressed)