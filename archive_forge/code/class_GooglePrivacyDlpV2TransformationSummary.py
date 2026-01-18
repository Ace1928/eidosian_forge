from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationSummary(_messages.Message):
    """Summary of a single transformation. Only one of 'transformation',
  'field_transformation', or 'record_suppress' will be set.

  Fields:
    field: Set if the transformation was limited to a specific FieldId.
    fieldTransformations: The field transformation that was applied. If
      multiple field transformations are requested for a single field, this
      list will contain all of them; otherwise, only one is supplied.
    infoType: Set if the transformation was limited to a specific InfoType.
    recordSuppress: The specific suppression option these stats apply to.
    results: Collection of all transformations that took place or had an
      error.
    transformation: The specific transformation these stats apply to.
    transformedBytes: Total size in bytes that were transformed in some way.
  """
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)
    fieldTransformations = _messages.MessageField('GooglePrivacyDlpV2FieldTransformation', 2, repeated=True)
    infoType = _messages.MessageField('GooglePrivacyDlpV2InfoType', 3)
    recordSuppress = _messages.MessageField('GooglePrivacyDlpV2RecordSuppression', 4)
    results = _messages.MessageField('GooglePrivacyDlpV2SummaryResult', 5, repeated=True)
    transformation = _messages.MessageField('GooglePrivacyDlpV2PrimitiveTransformation', 6)
    transformedBytes = _messages.IntegerField(7)