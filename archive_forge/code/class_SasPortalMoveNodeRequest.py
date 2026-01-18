from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalMoveNodeRequest(_messages.Message):
    """Request for MoveNode.

  Fields:
    destination: Required. The name of the new parent resource node or
      customer to reparent the node under.
  """
    destination = _messages.StringField(1)