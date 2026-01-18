from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RecordLocation(_messages.Message):
    """Location of a finding within a row or record.

  Fields:
    fieldId: Field id of the field containing the finding.
    recordKey: Key of the finding.
    tableLocation: Location within a `ContentItem.Table`.
  """
    fieldId = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)
    recordKey = _messages.MessageField('GooglePrivacyDlpV2RecordKey', 2)
    tableLocation = _messages.MessageField('GooglePrivacyDlpV2TableLocation', 3)