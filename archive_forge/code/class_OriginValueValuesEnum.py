from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OriginValueValuesEnum(_messages.Enum):
    """One of the standard Origins defined above.

    Values:
      SYSTEM: Counter was created by the Dataflow system.
      USER: Counter was created by the user.
    """
    SYSTEM = 0
    USER = 1