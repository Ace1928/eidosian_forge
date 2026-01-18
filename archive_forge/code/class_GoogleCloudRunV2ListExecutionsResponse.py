from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ListExecutionsResponse(_messages.Message):
    """Response message containing a list of Executions.

  Fields:
    executions: The resulting list of Executions.
    nextPageToken: A token indicating there are more items than page_size. Use
      it in the next ListExecutions request to continue.
  """
    executions = _messages.MessageField('GoogleCloudRunV2Execution', 1, repeated=True)
    nextPageToken = _messages.StringField(2)