from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiRegionMetadata(_messages.Message):
    """The metadata for the multi-region that includes the constituent regions.
  The metadata is only populated if the region is multi-region. For single
  region, it will be empty.

  Fields:
    constituentRegions: The regions constituting the multi-region.
  """
    constituentRegions = _messages.StringField(1, repeated=True)