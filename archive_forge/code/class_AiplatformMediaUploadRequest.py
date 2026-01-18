from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformMediaUploadRequest(_messages.Message):
    """A AiplatformMediaUploadRequest object.

  Fields:
    googleCloudAiplatformV1beta1UploadRagFileRequest: A
      GoogleCloudAiplatformV1beta1UploadRagFileRequest resource to be passed
      as the request body.
    parent: Required. The name of the RagCorpus resource into which to upload
      the file. Format:
      `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
  """
    googleCloudAiplatformV1beta1UploadRagFileRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1UploadRagFileRequest', 1)
    parent = _messages.StringField(2, required=True)