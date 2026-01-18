from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class BackendBuckets(base.Group):
    """Read and manipulate backend buckets.

  Backend buckets define Cloud Storage buckets that can serve content.
  URL maps define which requests are sent to which backend buckets. For more
  information, see:
  https://cloud.google.com/load-balancing/docs/https/ext-load-balancer-backend-buckets.
  """