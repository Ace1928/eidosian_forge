from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import text
import six
def ParseTrafficAllocations(args_allocations, split_method):
    """Parses the user-supplied allocations into a format acceptable by the API.

  Args:
    args_allocations: The raw allocations passed on the command line. A dict
      mapping version_id (str) to the allocation (float).
    split_method: Whether the traffic will be split by ip or cookie. This
      affects the format we specify the splits in.

  Returns:
    A dict mapping version id (str) to traffic split (float).

  Raises:
    ServicesSplitTrafficError: if the sum of traffic allocations is zero.
  """
    max_decimal_places = 2 if split_method == 'ip' else 3
    sum_of_splits = sum([float(s) for s in args_allocations.values()])
    err = ServicesSplitTrafficError('Cannot set traffic split to zero. If you would like a version to receive no traffic, send 100% of traffic to other versions or delete the service.')
    if sum_of_splits == 0.0:
        raise err
    allocations = {}
    for version, split in six.iteritems(args_allocations):
        allocation = float(split) / sum_of_splits
        allocation = round(allocation, max_decimal_places)
        if allocation == 0.0:
            raise err
        allocations[version] = allocation
    total_splits = round(sum(allocations.values()), max_decimal_places)
    difference = total_splits - 1.0
    max_split = max(allocations.values())
    for version, split in sorted(allocations.items()):
        if max_split == split:
            allocations[version] -= difference
            break
    return allocations