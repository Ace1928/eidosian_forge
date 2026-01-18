from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RoboModeValueValuesEnum(_messages.Enum):
    """The mode in which Robo should run. Most clients should allow the
    server to populate this field automatically.

    Values:
      ROBO_MODE_UNSPECIFIED: This means that the server should choose the
        mode. Recommended.
      ROBO_VERSION_1: Runs Robo in UIAutomator-only mode without app resigning
      ROBO_VERSION_2: Runs Robo in standard Espresso with UIAutomator fallback
    """
    ROBO_MODE_UNSPECIFIED = 0
    ROBO_VERSION_1 = 1
    ROBO_VERSION_2 = 2