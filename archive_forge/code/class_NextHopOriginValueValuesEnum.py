from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NextHopOriginValueValuesEnum(_messages.Enum):
    """[Output Only] Indicates the origin of the route. Can be IGP (Interior
    Gateway Protocol), EGP (Exterior Gateway Protocol), or INCOMPLETE.

    Values:
      EGP: <no description>
      IGP: <no description>
      INCOMPLETE: <no description>
    """
    EGP = 0
    IGP = 1
    INCOMPLETE = 2