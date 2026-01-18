from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColorValueValuesEnum(_messages.Enum):
    """The state color for this threshold. Color is not allowed in a XyChart.

    Values:
      COLOR_UNSPECIFIED: Color is unspecified. Not allowed in well-formed
        requests.
      YELLOW: Crossing the threshold is "concerning" behavior.
      RED: Crossing the threshold is "emergency" behavior.
    """
    COLOR_UNSPECIFIED = 0
    YELLOW = 1
    RED = 2