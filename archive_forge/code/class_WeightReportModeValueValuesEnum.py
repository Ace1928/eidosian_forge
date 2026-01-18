from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WeightReportModeValueValuesEnum(_messages.Enum):
    """Weight report mode. used for weighted Load Balancing.

    Values:
      DISABLE: Health Checker will not parse the header field.
      DRY_RUN: Health Checker will parse and report the weight in the header
        field, but load balancing will not be based on the weights and will
        use equal weights.
      ENABLE: Health Checker will try to parse and report the weight in the
        header field, and load balancing will be based on the weights as long
        as all backends have a valid weight or only a subset of backends has
        the UNAVAILABLE_WEIGHT WeightError. The latter case is to continue the
        weighted load balancing while some backends are in TIMEOUT or UNKNOWN
        health status.
    """
    DISABLE = 0
    DRY_RUN = 1
    ENABLE = 2