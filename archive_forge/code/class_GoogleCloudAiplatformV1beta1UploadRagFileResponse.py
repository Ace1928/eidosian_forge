from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UploadRagFileResponse(_messages.Message):
    """Response message for VertexRagDataService.UploadRagFile.

  Fields:
    error: The error that occurred while processing the RagFile.
    ragFile: The RagFile that had been uploaded into the RagCorpus.
  """
    error = _messages.MessageField('GoogleRpcStatus', 1)
    ragFile = _messages.MessageField('GoogleCloudAiplatformV1beta1RagFile', 2)