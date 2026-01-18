from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ScannedData(_messages.Message):
    """The data scanned during processing (e.g. in incremental DataScan)

  Fields:
    incrementalField: The range denoted by values of an incremental field
  """
    incrementalField = _messages.MessageField('GoogleCloudDataplexV1ScannedDataIncrementalField', 1)