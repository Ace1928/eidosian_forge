from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IpMapping(_messages.Message):
    """Database instance IP mapping

  Enums:
    TypeValueValuesEnum: The type of this IP address. A `PRIMARY` address is a
      public address that can accept incoming connections. A `PRIVATE` address
      is a private address that can accept incoming connections. An `OUTGOING`
      address is the source address of connections originating from the
      instance, if supported.

  Fields:
    ipAddress: The IP address assigned.
    timeToRetire: The due time for this IP to be retired in [RFC
      3339](https://tools.ietf.org/html/rfc3339) format, for example
      `2012-11-15T16:19:00.094Z`. This field is only available when the IP is
      scheduled to be retired.
    type: The type of this IP address. A `PRIMARY` address is a public address
      that can accept incoming connections. A `PRIVATE` address is a private
      address that can accept incoming connections. An `OUTGOING` address is
      the source address of connections originating from the instance, if
      supported.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of this IP address. A `PRIMARY` address is a public address
    that can accept incoming connections. A `PRIVATE` address is a private
    address that can accept incoming connections. An `OUTGOING` address is the
    source address of connections originating from the instance, if supported.

    Values:
      SQL_IP_ADDRESS_TYPE_UNSPECIFIED: This is an unknown IP address type.
      PRIMARY: IP address the customer is supposed to connect to. Usually this
        is the load balancer's IP address
      OUTGOING: Source IP address of the connection a read replica establishes
        to its external primary instance. This IP address can be allowlisted
        by the customer in case it has a firewall that filters incoming
        connection to its on premises primary instance.
      PRIVATE: Private IP used when using private IPs and network peering.
      MIGRATED_1ST_GEN: V1 IP of a migrated instance. We want the user to
        decommission this IP as soon as the migration is complete. Note: V1
        instances with V1 ip addresses will be counted as PRIMARY.
    """
        SQL_IP_ADDRESS_TYPE_UNSPECIFIED = 0
        PRIMARY = 1
        OUTGOING = 2
        PRIVATE = 3
        MIGRATED_1ST_GEN = 4
    ipAddress = _messages.StringField(1)
    timeToRetire = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)