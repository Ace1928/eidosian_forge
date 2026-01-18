from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchWriteResponse(_messages.Message):
    """The response from Firestore.BatchWrite.

  Fields:
    status: The status of applying the writes. This i-th write status
      corresponds to the i-th write in the request.
    writeResults: The result of applying the writes. This i-th write result
      corresponds to the i-th write in the request.
  """
    status = _messages.MessageField('Status', 1, repeated=True)
    writeResults = _messages.MessageField('WriteResult', 2, repeated=True)