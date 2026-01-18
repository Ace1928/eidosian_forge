from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvailabilityConfiguration(_messages.Message):
    """Configuration for availability of database instance

  Enums:
    AvailabilityTypeValueValuesEnum: Availability type. Potential values: *
      `ZONAL`: The instance serves data from only one zone. Outages in that
      zone affect data accessibility. * `REGIONAL`: The instance can serve
      data from more than one zone in a region (it is highly available).

  Fields:
    availabilityType: Availability type. Potential values: * `ZONAL`: The
      instance serves data from only one zone. Outages in that zone affect
      data accessibility. * `REGIONAL`: The instance can serve data from more
      than one zone in a region (it is highly available).
    externalReplicaConfigured: A boolean attribute.
    promotableReplicaConfigured: A boolean attribute.
  """

    class AvailabilityTypeValueValuesEnum(_messages.Enum):
        """Availability type. Potential values: * `ZONAL`: The instance serves
    data from only one zone. Outages in that zone affect data accessibility. *
    `REGIONAL`: The instance can serve data from more than one zone in a
    region (it is highly available).

    Values:
      AVAILABILITY_TYPE_UNSPECIFIED: <no description>
      ZONAL: Zonal available instance.
      REGIONAL: Regional available instance.
      MULTI_REGIONAL: Multi regional instance
      AVAILABILITY_TYPE_OTHER: For rest of the other category
    """
        AVAILABILITY_TYPE_UNSPECIFIED = 0
        ZONAL = 1
        REGIONAL = 2
        MULTI_REGIONAL = 3
        AVAILABILITY_TYPE_OTHER = 4
    availabilityType = _messages.EnumField('AvailabilityTypeValueValuesEnum', 1)
    externalReplicaConfigured = _messages.BooleanField(2)
    promotableReplicaConfigured = _messages.BooleanField(3)