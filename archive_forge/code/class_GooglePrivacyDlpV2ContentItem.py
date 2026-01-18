from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ContentItem(_messages.Message):
    """Type of content to inspect.

  Fields:
    byteItem: Content data to inspect or redact. Replaces `type` and `data`.
    table: Structured content for inspection. See
      https://cloud.google.com/sensitive-data-protection/docs/inspecting-
      text#inspecting_a_table to learn more.
    value: String data to inspect or redact.
  """
    byteItem = _messages.MessageField('GooglePrivacyDlpV2ByteContentItem', 1)
    table = _messages.MessageField('GooglePrivacyDlpV2Table', 2)
    value = _messages.StringField(3)