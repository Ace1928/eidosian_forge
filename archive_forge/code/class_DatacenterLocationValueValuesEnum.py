from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacenterLocationValueValuesEnum(_messages.Enum):
    """Output only. Datacenter location used to register the session.

    Values:
      DATACENTER_LOCATION_UNSPECIFIED: Default value. Should not be used.
      US: US datacenter location.
      EU: EU datacenter location.
      SGP: Singapore datacenter location.
      SIN: Singapore datacenter location.
      UK: UK datacenter location.
      JPN: Japan datacenter location.
      CAN: Canada(Montr\\xe9al) data center.
    """
    DATACENTER_LOCATION_UNSPECIFIED = 0
    US = 1
    EU = 2
    SGP = 3
    SIN = 4
    UK = 5
    JPN = 6
    CAN = 7