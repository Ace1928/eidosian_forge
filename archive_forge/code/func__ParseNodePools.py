from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
def _ParseNodePools(self, dataproc, args_node_pools):
    """Parses all --node-pool flags into a list of NodePool messages."""
    pools = [self._ParseNodePool(dataproc, node_pool) for node_pool in args_node_pools]
    self._ValidateNodePoolIds(dataproc, pools)
    return pools