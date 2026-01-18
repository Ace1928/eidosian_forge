from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CapacityUnitsValueValuesEnum(_messages.Enum):
    """CapacityUnitsValueValuesEnum enum type.

    Values:
      CAPACITY_UNITS_UNSPECIFIED: The capacity units is not known/set.
      CORES: The capacity unit is set to CORES.
      CHIPS: The capacity unit is set to CHIPS.
    """
    CAPACITY_UNITS_UNSPECIFIED = 0
    CORES = 1
    CHIPS = 2