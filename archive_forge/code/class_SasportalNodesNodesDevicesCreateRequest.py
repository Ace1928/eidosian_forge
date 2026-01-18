from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalNodesNodesDevicesCreateRequest(_messages.Message):
    """A SasportalNodesNodesDevicesCreateRequest object.

  Fields:
    parent: Required. The name of the parent resource.
    sasPortalDevice: A SasPortalDevice resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    sasPortalDevice = _messages.MessageField('SasPortalDevice', 2)