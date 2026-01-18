from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def ListKrmApiHosts(parent):
    """Calls into the ListKrmApiHosts API.

  Args:
    parent: the fully qualified name of the parent, e.g.
      "projects/p/locations/l".

  Returns:
    A list of messages.KrmApiHost.
  """
    client = GetClientInstance()
    messages = client.MESSAGES_MODULE
    resp = client.projects_locations_krmApiHosts.List(messages.KrmapihostingProjectsLocationsKrmApiHostsListRequest(parent=parent))
    return resp.krmApiHosts