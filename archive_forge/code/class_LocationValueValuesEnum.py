from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationValueValuesEnum(_messages.Enum):
    """The location where this mapping applies.

    Values:
      UNKNOWN: <no description>
      PATH: <no description>
      QUERY: <no description>
      BODY: <no description>
      HEADER: <no description>
    """
    UNKNOWN = 0
    PATH = 1
    QUERY = 2
    BODY = 3
    HEADER = 4