from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SingleRegionQuorum(_messages.Message):
    """Message type for a single-region quorum.

  Fields:
    servingLocation: Required. The location of the serving region, e.g. "us-
      central1". The location must be one of the regions within the dual
      region instance configuration of your database. The list of valid
      locations is available via
      [GetInstanceConfig[InstanceAdmin.GetInstanceConfig] API. This should
      only be used if you plan to change quorum in single-region quorum type.
  """
    servingLocation = _messages.StringField(1)