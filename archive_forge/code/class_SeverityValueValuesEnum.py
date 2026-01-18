from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeverityValueValuesEnum(_messages.Enum):
    """Severity of issue. Required.

    Values:
      unspecifiedSeverity: Default unspecified severity. Do not use. For
        versioning only.
      info: Non critical issue, providing users with some info about the test
        run.
      suggestion: Non critical issue, providing users with some hints on
        improving their testing experience, e.g., suggesting to use Game
        Loops.
      warning: Potentially critical issue.
      severe: Critical issue.
    """
    unspecifiedSeverity = 0
    info = 1
    suggestion = 2
    warning = 3
    severe = 4