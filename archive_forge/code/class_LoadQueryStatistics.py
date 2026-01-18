from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadQueryStatistics(_messages.Message):
    """Statistics for a LOAD query.

  Fields:
    badRecords: Output only. The number of bad records encountered while
      processing a LOAD query. Note that if the job has failed because of more
      bad records encountered than the maximum allowed in the load job
      configuration, then this number can be less than the total number of bad
      records present in the input data.
    bytesTransferred: Output only. This field is deprecated. The number of
      bytes of source data copied over the network for a `LOAD` query.
      `transferred_bytes` has the canonical value for physical transferred
      bytes, which is used for BigQuery Omni billing.
    inputFileBytes: Output only. Number of bytes of source data in a LOAD
      query.
    inputFiles: Output only. Number of source files in a LOAD query.
    outputBytes: Output only. Size of the loaded data in bytes. Note that
      while a LOAD query is in the running state, this value may change.
    outputRows: Output only. Number of rows imported in a LOAD query. Note
      that while a LOAD query is in the running state, this value may change.
  """
    badRecords = _messages.IntegerField(1)
    bytesTransferred = _messages.IntegerField(2)
    inputFileBytes = _messages.IntegerField(3)
    inputFiles = _messages.IntegerField(4)
    outputBytes = _messages.IntegerField(5)
    outputRows = _messages.IntegerField(6)