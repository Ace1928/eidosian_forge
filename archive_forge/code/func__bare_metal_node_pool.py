from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import bare_metal_clusters as clusters
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _bare_metal_node_pool(self, args: parser_extensions.Namespace):
    """Constructs proto message BareMetalNodePool."""
    kwargs = {'name': self._node_pool_name(args), 'nodePoolConfig': self._node_pool_config(args), 'displayName': getattr(args, 'display_name', None), 'annotations': self._annotations(args), 'bareMetalVersion': getattr(args, 'version', None)}
    return messages.BareMetalNodePool(**kwargs)