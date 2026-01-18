from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _node_taints(self, args: parser_extensions.Namespace):
    taint_messages = []
    node_taints = flags.Get(args, 'node_taints', {})
    for node_taint in node_taints.items():
        taint_object = self._parse_node_taint(node_taint)
        taint_messages.append(messages.NodeTaint(**taint_object))
    return taint_messages