from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryField(_messages.Message):
    """Message defining a field of a BigQuery table.

  Fields:
    field: Designated field in the BigQuery table.
    table: Source table of the field.
  """
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)
    table = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 2)