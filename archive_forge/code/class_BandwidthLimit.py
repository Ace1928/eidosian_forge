from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BandwidthLimit(_messages.Message):
    """Specifies a bandwidth limit for an agent pool.

  Fields:
    limitMbps: Bandwidth rate in megabytes per second, distributed across all
      the agents in the pool.
  """
    limitMbps = _messages.IntegerField(1)