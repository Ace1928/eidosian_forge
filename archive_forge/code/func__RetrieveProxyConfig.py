from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import re
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util as hub_kube_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _RetrieveProxyConfig(is_mcp, mesh_config):
    """Retrieve proxy config from a mesh config.

  Args:
    is_mcp: Whether the control plane is managed or not.
    mesh_config: A mesh config from the cluster.

  Returns:
    proxy_config: The proxy config from the mesh config.
  """
    try:
        proxy_config = mesh_config['defaultConfig']
    except (KeyError, TypeError):
        if is_mcp:
            return {}
        raise exceptions.Error('Proxy config cannot be found in the Anthos Service Mesh.')
    return proxy_config