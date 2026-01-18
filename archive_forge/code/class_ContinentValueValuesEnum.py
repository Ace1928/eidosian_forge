from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContinentValueValuesEnum(_messages.Enum):
    """[Output Only] Continent for this location, which can take one of the
    following values: - AFRICA - ASIA_PAC - EUROPE - NORTH_AMERICA -
    SOUTH_AMERICA

    Values:
      AFRICA: <no description>
      ASIA_PAC: <no description>
      EUROPE: <no description>
      NORTH_AMERICA: <no description>
      SOUTH_AMERICA: <no description>
    """
    AFRICA = 0
    ASIA_PAC = 1
    EUROPE = 2
    NORTH_AMERICA = 3
    SOUTH_AMERICA = 4