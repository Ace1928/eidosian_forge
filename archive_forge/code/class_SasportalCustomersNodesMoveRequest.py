from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersNodesMoveRequest(_messages.Message):
    """A SasportalCustomersNodesMoveRequest object.

  Fields:
    name: Required. The name of the node to move.
    sasPortalMoveNodeRequest: A SasPortalMoveNodeRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    sasPortalMoveNodeRequest = _messages.MessageField('SasPortalMoveNodeRequest', 2)