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
def GetScopeNamespace(self, namespace_path):
    """Gets a namespace resource from the GKEHub API.

    Args:
      namespace_path: Full resource path of the namespace.

    Returns:
      A namespace resource.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsScopesNamespacesGetRequest(name=namespace_path)
    return self.client.projects_locations_scopes_namespaces.Get(req)