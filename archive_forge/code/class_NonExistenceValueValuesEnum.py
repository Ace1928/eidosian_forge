from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class NonExistenceValueValuesEnum(_messages.Enum):
    """Specifies the mechanism for authenticated denial-of-existence
    responses. Can only be changed while the state is OFF.

    Values:
      nsec: <no description>
      nsec3: <no description>
    """
    nsec = 0
    nsec3 = 1