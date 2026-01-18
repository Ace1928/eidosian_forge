from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Keystore(_messages.Message):
    """Datastore for Certificates and Aliases.

  Fields:
    aliases: Output only. Aliases in this keystore.
    name: Required. Resource ID for this keystore. Values must match the
      regular expression `[\\w[:space:].-]{1,255}`.
  """
    aliases = _messages.StringField(1, repeated=True)
    name = _messages.StringField(2)