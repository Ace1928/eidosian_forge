from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerdictValueValuesEnum(_messages.Enum):
    """Verdict indicates the assessment result.

    Values:
      VERDICT_UNSPECIFIED: The verdict is unspecified.
      PASS: The assessment has passed.
      FAIL: The assessment has failed.
    """
    VERDICT_UNSPECIFIED = 0
    PASS = 1
    FAIL = 2