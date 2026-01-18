from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigFoldersOsConfigsCreateRequest(_messages.Message):
    """A OsconfigFoldersOsConfigsCreateRequest object.

  Fields:
    osConfig: A OsConfig resource to be passed as the request body.
    parent: The resource name of the parent.
  """
    osConfig = _messages.MessageField('OsConfig', 1)
    parent = _messages.StringField(2, required=True)