from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalMoveDeploymentRequest(_messages.Message):
    """Request for MoveDeployment.

  Fields:
    destination: Required. The name of the new parent resource node or
      customer to reparent the deployment under.
  """
    destination = _messages.StringField(1)