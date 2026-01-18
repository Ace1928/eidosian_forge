from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContinuousAnalysisValueValuesEnum(_messages.Enum):
    """Whether the resource is continuously analyzed.

    Values:
      CONTINUOUS_ANALYSIS_UNSPECIFIED: Unknown.
      ACTIVE: The resource is continuously analyzed.
      INACTIVE: The resource is ignored for continuous analysis.
    """
    CONTINUOUS_ANALYSIS_UNSPECIFIED = 0
    ACTIVE = 1
    INACTIVE = 2