from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MeasurementCapabilitiesValueListEntryValuesEnum(_messages.Enum):
    """MeasurementCapabilitiesValueListEntryValuesEnum enum type.

    Values:
      MEASUREMENT_CAPABILITY_UNSPECIFIED: <no description>
      MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITH_GRANT: <no description>
      MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITHOUT_GRANT: <no description>
    """
    MEASUREMENT_CAPABILITY_UNSPECIFIED = 0
    MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITH_GRANT = 1
    MEASUREMENT_CAPABILITY_RECEIVED_POWER_WITHOUT_GRANT = 2