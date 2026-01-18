from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ZoneResourceSpec(_messages.Message):
    """Settings for resources attached as assets within a zone.

  Enums:
    LocationTypeValueValuesEnum: Required. Immutable. The location type of the
      resources that are allowed to be attached to the assets within this
      zone.

  Fields:
    locationType: Required. Immutable. The location type of the resources that
      are allowed to be attached to the assets within this zone.
  """

    class LocationTypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. The location type of the resources that are
    allowed to be attached to the assets within this zone.

    Values:
      LOCATION_TYPE_UNSPECIFIED: Unspecified location type.
      SINGLE_REGION: Resources that are associated with a single region.
      MULTI_REGION: Resources that are associated with a multi-region
        location.
    """
        LOCATION_TYPE_UNSPECIFIED = 0
        SINGLE_REGION = 1
        MULTI_REGION = 2
    locationType = _messages.EnumField('LocationTypeValueValuesEnum', 1)