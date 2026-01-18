from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p1beta1AsyncBatchAnnotateFilesResponse(_messages.Message):
    """Response to an async batch file annotation request.

  Fields:
    responses: The list of file annotation responses, one for each request in
      AsyncBatchAnnotateFilesRequest.
  """
    responses = _messages.MessageField('GoogleCloudVisionV1p1beta1AsyncAnnotateFileResponse', 1, repeated=True)