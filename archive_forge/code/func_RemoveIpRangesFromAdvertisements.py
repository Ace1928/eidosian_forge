from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def RemoveIpRangesFromAdvertisements(messages, resource_class, resource, ip_ranges):
    """Removes all specified IP ranges from a resource's advertisements.

  Raises an error if any of the specified advertised IP ranges were not found in
  the resource's advertisement set. The IP range search is done by exact text
  match (ignoring descriptions).

  Args:
    messages: API messages holder.
    resource_class: RouterBgp or RouterBgpPeer class type being modified.
    resource: the resource (router/peer) being modified.
    ip_ranges: the advertised IP ranges to remove.

  Raises:
    IpRangeNotFoundError: if any IP range was not found in the resource.
  """
    for ip_range in ip_ranges:
        if ip_range not in [r.range for r in resource.advertisedIpRanges]:
            raise IpRangeNotFoundError(messages, resource_class, _ADVERTISED_IP_RANGE_NOT_FOUND_ERROR_MESSAGE, ip_range)
    resource.advertisedIpRanges = [r for r in resource.advertisedIpRanges if r.range not in ip_ranges]