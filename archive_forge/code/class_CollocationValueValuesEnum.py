from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CollocationValueValuesEnum(_messages.Enum):
    """Specifies network collocation

    Values:
      COLLOCATED: <no description>
      UNSPECIFIED_COLLOCATION: <no description>
    """
    COLLOCATED = 0
    UNSPECIFIED_COLLOCATION = 1