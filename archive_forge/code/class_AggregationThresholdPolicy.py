from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationThresholdPolicy(_messages.Message):
    """Represents privacy policy associated with "aggregation threshold"
  method.

  Fields:
    privacyUnitColumns: Optional. The privacy unit column(s) associated with
      this policy. For now, only one column per data source object (table,
      view) is allowed as a privacy unit column. Representing as a repeated
      field in metadata for extensibility to multiple columns in future.
      Duplicates and Repeated struct fields are not allowed. For nested
      fields, use dot notation ("outer.inner")
    threshold: Optional. The threshold for the "aggregation threshold" policy.
  """
    privacyUnitColumns = _messages.StringField(1, repeated=True)
    threshold = _messages.IntegerField(2)