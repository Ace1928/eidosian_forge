from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScaleTypeValueValuesEnum(_messages.Enum):
    """How the parameter should be scaled. Leave unset for `CATEGORICAL`
    parameters.

    Values:
      SCALE_TYPE_UNSPECIFIED: By default, no scaling is applied.
      UNIT_LINEAR_SCALE: Scales the feasible space to (0, 1) linearly.
      UNIT_LOG_SCALE: Scales the feasible space logarithmically to (0, 1). The
        entire feasible space must be strictly positive.
      UNIT_REVERSE_LOG_SCALE: Scales the feasible space "reverse"
        logarithmically to (0, 1). The result is that values close to the top
        of the feasible space are spread out more than points near the bottom.
        The entire feasible space must be strictly positive.
    """
    SCALE_TYPE_UNSPECIFIED = 0
    UNIT_LINEAR_SCALE = 1
    UNIT_LOG_SCALE = 2
    UNIT_REVERSE_LOG_SCALE = 3