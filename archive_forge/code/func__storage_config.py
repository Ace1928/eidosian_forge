from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _storage_config(self, args: parser_extensions.Namespace):
    """Constructs proto message BareMetalAdminStorageConfig."""
    kwargs = {'lvpShareConfig': self._lvp_share_config(args), 'lvpNodeMountsConfig': self._lvp_node_mounts_config(args)}
    if any(kwargs.values()):
        return messages.BareMetalAdminStorageConfig(**kwargs)
    return None