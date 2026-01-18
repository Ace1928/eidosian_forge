from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AssetDiscoverySpecCsvOptions(_messages.Message):
    """Describe CSV and similar semi-structured data formats.

  Fields:
    delimiter: Optional. The delimiter being used to separate values. This
      defaults to ','.
    disableTypeInference: Optional. Whether to disable the inference of data
      type for CSV data. If true, all columns will be registered as strings.
    encoding: Optional. The character encoding of the data. The default is
      UTF-8.
    headerRows: Optional. The number of rows to interpret as header rows that
      should be skipped when reading data rows.
  """
    delimiter = _messages.StringField(1)
    disableTypeInference = _messages.BooleanField(2)
    encoding = _messages.StringField(3)
    headerRows = _messages.IntegerField(4, variant=_messages.Variant.INT32)