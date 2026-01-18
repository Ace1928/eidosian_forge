from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationalStatusValueValuesEnum(_messages.Enum):
    """The operational status of the link.

    Values:
      LINK_OPERATIONAL_STATUS_DOWN: The interface is unable to communicate
        with the remote end.
      LINK_OPERATIONAL_STATUS_UP: The interface has low level communication
        with the remote end.
    """
    LINK_OPERATIONAL_STATUS_DOWN = 0
    LINK_OPERATIONAL_STATUS_UP = 1