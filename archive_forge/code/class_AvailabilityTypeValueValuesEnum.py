from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvailabilityTypeValueValuesEnum(_messages.Enum):
    """Optional. Availability type. Potential values: * `ZONAL`: The instance
    serves data from only one zone. Outages in that zone affect data
    availability. * `REGIONAL`: The instance can serve data from more than one
    zone in a region (it is highly available).

    Values:
      SQL_AVAILABILITY_TYPE_UNSPECIFIED: This is an unknown Availability type.
      ZONAL: Zonal availablility instance.
      REGIONAL: Regional availability instance.
    """
    SQL_AVAILABILITY_TYPE_UNSPECIFIED = 0
    ZONAL = 1
    REGIONAL = 2