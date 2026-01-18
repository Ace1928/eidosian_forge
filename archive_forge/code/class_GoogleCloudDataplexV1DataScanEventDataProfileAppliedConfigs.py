from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanEventDataProfileAppliedConfigs(_messages.Message):
    """Applied configs for data profile type data scan job.

  Fields:
    columnFilterApplied: Boolean indicating whether a column filter was
      applied in the DataScan job.
    rowFilterApplied: Boolean indicating whether a row filter was applied in
      the DataScan job.
    samplingPercent: The percentage of the records selected from the dataset
      for DataScan. Value ranges between 0.0 and 100.0. Value 0.0 or 100.0
      imply that sampling was not applied.
  """
    columnFilterApplied = _messages.BooleanField(1)
    rowFilterApplied = _messages.BooleanField(2)
    samplingPercent = _messages.FloatField(3, variant=_messages.Variant.FLOAT)