from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RecordTransformations(_messages.Message):
    """A type of transformation that is applied over structured data such as a
  table.

  Fields:
    fieldTransformations: Transform the record by applying various field
      transformations.
    recordSuppressions: Configuration defining which records get suppressed
      entirely. Records that match any suppression rule are omitted from the
      output.
  """
    fieldTransformations = _messages.MessageField('GooglePrivacyDlpV2FieldTransformation', 1, repeated=True)
    recordSuppressions = _messages.MessageField('GooglePrivacyDlpV2RecordSuppression', 2, repeated=True)