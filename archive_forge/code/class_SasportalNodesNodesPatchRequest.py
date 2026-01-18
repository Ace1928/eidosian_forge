from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalNodesNodesPatchRequest(_messages.Message):
    """A SasportalNodesNodesPatchRequest object.

  Fields:
    name: Output only. Resource name.
    sasPortalNode: A SasPortalNode resource to be passed as the request body.
    updateMask: Fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    sasPortalNode = _messages.MessageField('SasPortalNode', 2)
    updateMask = _messages.StringField(3)