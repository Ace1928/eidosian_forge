from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchAnnotateFilesResponse(_messages.Message):
    """A list of file annotation responses.

  Fields:
    responses: The list of file annotation responses, each response
      corresponding to each AnnotateFileRequest in BatchAnnotateFilesRequest.
  """
    responses = _messages.MessageField('AnnotateFileResponse', 1, repeated=True)