from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1RawDocument(_messages.Message):
    """Payload message of raw document content (bytes).

  Fields:
    content: Inline document content.
    displayName: The display name of the document, it supports all Unicode
      characters except the following: `*`, `?`, `[`, `]`, `%`, `{`, `}`,`'`,
      `\\"`, `,` `~`, `=` and `:` are reserved. If not specified, a default ID
      is generated.
    mimeType: An IANA MIME type (RFC6838) indicating the nature and format of
      the content.
  """
    content = _messages.BytesField(1)
    displayName = _messages.StringField(2)
    mimeType = _messages.StringField(3)