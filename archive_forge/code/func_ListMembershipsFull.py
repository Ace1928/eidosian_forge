from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
import six
def ListMembershipsFull(filter_cluster_missing=False):
    """Lists full Membership names in the fleet for the current project.

  Args:
    filter_cluster_missing: whether to filter out memberships that are missing
    a cluster.
  Returns:
    A list of full membership resource names in the fleet in the form
    'projects/*/locations/*/memberships/*'.
    A list of locations which were unreachable.
  """
    client = core_apis.GetClientInstance('gkehub', 'v1beta1')
    req = client.MESSAGES_MODULE.GkehubProjectsLocationsMembershipsListRequest(parent=hub_base.HubCommand.LocationResourceName(location='-'))
    unreachable = set()

    def _GetFieldFunc(message, attribute):
        unreachable.update(message.unreachable)
        return getattr(message, attribute)
    result = list_pager.YieldFromList(client.projects_locations_memberships, req, field='resources', batch_size_attribute=None, get_field_func=_GetFieldFunc)
    if filter_cluster_missing:
        memberships = [m.name for m in result if not _ClusterMissing(m.endpoint)]
    else:
        memberships = [m.name for m in result]
    return (memberships, list(unreachable))