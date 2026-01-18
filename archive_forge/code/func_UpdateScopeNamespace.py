from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def UpdateScopeNamespace(self, namespace_path, labels=None, namespace_labels=None, mask=''):
    """Updates a namespace resource in the fleet.

    Args:
      namespace_path: Full resource path of the namespace.
      labels:  Labels for the resource.
      namespace_labels:  Namespace-level labels for the cluster namespace.
      mask: A mask of the fields to update.

    Returns:
      A longrunning operation for updating the namespace.

    Raises:
    """
    namespace = self.messages.Namespace(name=namespace_path, scope='', labels=labels, namespaceLabels=namespace_labels)
    req = self.messages.GkehubProjectsLocationsScopesNamespacesPatchRequest(namespace=namespace, name=namespace_path, updateMask=mask)
    op = self.client.projects_locations_scopes_namespaces.Patch(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be updated')