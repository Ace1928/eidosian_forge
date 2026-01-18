from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def RemoveGroupsFromAdvertisements(messages, resource_class, resource, groups):
    """Remove all specified groups from a resource's advertisements.

  Raises an error if any of the specified advertised groups were not found in
  the resource's advertisement set.

  Args:
    messages: API messages holder.
    resource_class: RouterBgp or RouterBgpPeer class type being modified.
    resource: the resource (router/peer) being modified.
    groups: the advertised groups to remove.

  Raises:
    GroupNotFoundError: if any group was not found in the resource.
  """
    for group in groups:
        if group not in resource.advertisedGroups:
            raise GroupNotFoundError(messages, resource_class, group)
    resource.advertisedGroups = [g for g in resource.advertisedGroups if g not in groups]