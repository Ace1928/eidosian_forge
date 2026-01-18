from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigFoldersOsConfigsPatchRequest(_messages.Message):
    """A OsconfigFoldersOsConfigsPatchRequest object.

  Fields:
    name: The resource name of the OsConfig.
    osConfig: A OsConfig resource to be passed as the request body.
    updateMask: Field mask that controls which fields of the OsConfig should
      be updated.
  """
    name = _messages.StringField(1, required=True)
    osConfig = _messages.MessageField('OsConfig', 2)
    updateMask = _messages.StringField(3)