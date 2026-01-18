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
def _ValidateNodePoolIds(self, dataproc, node_pools):
    """Validates whether node-pools are valid."""
    valid_ids = self._GetValidNodePoolIdChoices(dataproc)
    for node_pool in node_pools:
        node_pool_id = node_pool.id
        if node_pool_id not in valid_ids:
            raise exceptions.InvalidArgumentException('--node-pool', 'Node pool ID "{}" is not one of {}'.format(node_pool_id, valid_ids))
    unique_ids = set()
    for node_pool in node_pools:
        node_pool_id = node_pool.id
        if node_pool_id in unique_ids:
            raise exceptions.InvalidArgumentException('--node-pool', 'Node pool id "{}" used more than once.'.format(node_pool_id))
        unique_ids.add(node_pool_id)