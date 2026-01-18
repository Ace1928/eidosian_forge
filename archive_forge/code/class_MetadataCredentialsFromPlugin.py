from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataCredentialsFromPlugin(_messages.Message):
    """[Deprecated] Custom authenticator credentials. Custom authenticator
  credentials.

  Fields:
    name: Plugin name.
    structConfig: A text proto that conforms to a Struct type definition
      interpreted by the plugin.
  """
    name = _messages.StringField(1)
    structConfig = _messages.StringField(2)