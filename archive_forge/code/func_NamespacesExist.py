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
def NamespacesExist(self, *namespaces):
    """Check to see if the namespaces exist in the cluster.

    Args:
      *namespaces: The namespaces to check.

    Returns:
      true, if namespaces exist.

    Raises:
      Error: if failing to verify the namespaces.
      Error: if at least one of the namespaces do not exist.
    """
    for ns in namespaces:
        _, err = self._RunKubectl(['get', 'namespace', ns], None)
        if err:
            if 'NotFound' in err:
                raise exceptions.Error('Namespace {} does not exist: {}'.format(ns, err))
            raise exceptions.Error('Failed to check if namespace {} exists: {}'.format(ns, err))
    return True