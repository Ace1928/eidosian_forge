from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LevelValueValuesEnum(_messages.Enum):
    """Represents how severe a message is.

    Values:
      LEVEL_UNSPECIFIED: Illegal. Same
        istio.analysis.v1alpha1.AnalysisMessageBase.Level.UNKNOWN.
      ERROR: ERROR represents a misconfiguration that must be fixed.
      WARNING: WARNING represents a misconfiguration that should be fixed.
      INFO: INFO represents an informational finding.
    """
    LEVEL_UNSPECIFIED = 0
    ERROR = 1
    WARNING = 2
    INFO = 3