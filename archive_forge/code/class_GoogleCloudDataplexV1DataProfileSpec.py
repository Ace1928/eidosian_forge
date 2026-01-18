from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataProfileSpec(_messages.Message):
    """DataProfileScan related setting.

  Fields:
    excludeFields: Optional. The fields to exclude from data profile.If
      specified, the fields will be excluded from data profile, regardless of
      include_fields value.
    includeFields: Optional. The fields to include in data profile.If not
      specified, all fields at the time of profile scan job execution are
      included, except for ones listed in exclude_fields.
    postScanActions: Optional. Actions to take upon job completion..
    rowFilter: Optional. A filter applied to all rows in a single DataScan
      job. The filter needs to be a valid SQL expression for a WHERE clause in
      BigQuery standard SQL syntax. Example: col1 >= 0 AND col2 < 10
    samplingPercent: Optional. The percentage of the records to be selected
      from the dataset for DataScan. Value can range between 0.0 and 100.0
      with up to 3 significant decimal digits. Sampling is not applied if
      sampling_percent is not specified, 0 or 100.
  """
    excludeFields = _messages.MessageField('GoogleCloudDataplexV1DataProfileSpecSelectedFields', 1)
    includeFields = _messages.MessageField('GoogleCloudDataplexV1DataProfileSpecSelectedFields', 2)
    postScanActions = _messages.MessageField('GoogleCloudDataplexV1DataProfileSpecPostScanActions', 3)
    rowFilter = _messages.StringField(4)
    samplingPercent = _messages.FloatField(5, variant=_messages.Variant.FLOAT)