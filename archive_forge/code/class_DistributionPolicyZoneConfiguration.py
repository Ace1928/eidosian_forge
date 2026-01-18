from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DistributionPolicyZoneConfiguration(_messages.Message):
    """A DistributionPolicyZoneConfiguration object.

  Fields:
    zone: The URL of the zone. The zone must exist in the region where the
      managed instance group is located.
  """
    zone = _messages.StringField(1)