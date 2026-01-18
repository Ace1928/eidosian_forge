from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceSizeValueValuesEnum(_messages.Enum):
    """An enum of readable instance sizes, with each instance size mapping to
    a float value (e.g. InstanceSize.EXTRA_SMALL = scaling_factor(0.1))

    Values:
      INSTANCE_SIZE_UNSPECIFIED: Unspecified instance size
      EXTRA_SMALL: Extra small instance size, maps to a scaling factor of 0.1.
      SMALL: Small instance size, maps to a scaling factor of 0.5.
      MEDIUM: Medium instance size, maps to a scaling factor of 1.0.
      LARGE: Large instance size, maps to a scaling factor of 3.0.
      EXTRA_LARGE: Extra large instance size, maps to a scaling factor of 6.0.
    """
    INSTANCE_SIZE_UNSPECIFIED = 0
    EXTRA_SMALL = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    EXTRA_LARGE = 5