from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigFoldersOsConfigsGetRequest(_messages.Message):
    """A OsconfigFoldersOsConfigsGetRequest object.

  Fields:
    name: The resource name of the OsConfig.
  """
    name = _messages.StringField(1, required=True)