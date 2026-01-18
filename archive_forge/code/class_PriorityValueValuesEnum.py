from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PriorityValueValuesEnum(_messages.Enum):
    """Relative importance of a suggestion. Always set.

    Values:
      unknownPriority: <no description>
      error: <no description>
      warning: <no description>
      info: <no description>
    """
    unknownPriority = 0
    error = 1
    warning = 2
    info = 3