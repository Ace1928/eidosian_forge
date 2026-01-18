from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UploadRagFileRequest(_messages.Message):
    """Request message for VertexRagDataService.UploadRagFile.

  Fields:
    ragFile: Required. The RagFile to upload.
    uploadRagFileConfig: Required. The config for the RagFiles to be uploaded
      into the RagCorpus. VertexRagDataService.UploadRagFile.
  """
    ragFile = _messages.MessageField('GoogleCloudAiplatformV1beta1RagFile', 1)
    uploadRagFileConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1UploadRagFileConfig', 2)