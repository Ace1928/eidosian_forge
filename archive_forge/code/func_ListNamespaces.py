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
def ListNamespaces(self, project):
    """Lists namespaces in a project.

    Args:
      project: the project to list namespaces from.

    Returns:
      A ListNamespaceResponse (list of namespaces and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    req = self.messages.GkehubProjectsLocationsNamespacesListRequest(pageToken='', parent=util.NamespaceParentName(project))
    return list_pager.YieldFromList(self.client.projects_locations_namespaces, req, field='namespaces', batch_size_attribute=None)