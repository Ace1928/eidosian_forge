from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GetClusterFallthrough():
    """Python hook to get the value for the '-' cluster.

  See details at:

  https://cloud.google.com/apis/design/design_patterns#list_sub-collections

  This allows us to operate on node pools without needing to specify a specific
  parent cluster.

  Returns:
    The value of the wildcard cluster.
  """
    return '-'