from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1RunDataScanResponse(_messages.Message):
    """Run DataScan Response.

  Fields:
    job: DataScanJob created by RunDataScan request.
  """
    job = _messages.MessageField('GoogleCloudDataplexV1DataScanJob', 1)