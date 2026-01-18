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
def CreateScopeNamespace(self, name, namespace_path, parent, labels=None, namespace_labels=None):
    """Creates a namespace resource from the GKEHub API.

    Args:
      name: The namespace name.
      namespace_path: Full resource path of the namespace.
      parent: Full resource path of the scope containing the namespace.
      labels: labels for namespace resource.
      namespace_labels: Namespace-level labels for the cluster namespace.

    Returns:
      A namespace resource.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error.
    """
    namespace = self.messages.Namespace(name=namespace_path, scope='', labels=labels, namespaceLabels=namespace_labels)
    req = self.messages.GkehubProjectsLocationsScopesNamespacesCreateRequest(namespace=namespace, scopeNamespaceId=name, parent=parent)
    op = self.client.projects_locations_scopes_namespaces.Create(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be created')