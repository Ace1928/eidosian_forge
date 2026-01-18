from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from six.moves import map  # pylint: disable=redefined-builtin
def ParseIpRanges(messages, ip_ranges):
    """Parses a dict of IP ranges into AdvertisedIpRange objects.

  Args:
    messages: API messages holder.
    ip_ranges: A dict of IP ranges of the form ip_range=description, where
      ip_range is a CIDR-formatted IP and description is an optional text label.

  Returns:
    A list of AdvertisedIpRange objects containing the specified IP ranges.
  """
    ranges = [messages.RouterAdvertisedIpRange(range=ip_range, description=description) for ip_range, description in ip_ranges.items()]
    ranges.sort(key=operator.attrgetter('range', 'description'))
    return ranges