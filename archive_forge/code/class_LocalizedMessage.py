from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalizedMessage(_messages.Message):
    """Provides a localized error message that is safe to return to the user
  which can be attached to an RPC error.

  Fields:
    locale: The locale used following the specification defined at
      https://www.rfc-editor.org/rfc/bcp/bcp47.txt. Examples are: "en-US",
      "fr-CH", "es-MX"
    message: The localized error message in the above locale.
  """
    locale = _messages.StringField(1)
    message = _messages.StringField(2)