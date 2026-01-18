from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaRagFilesImportRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaRagFilesImportRequest object.

  Fields:
    googleCloudAiplatformV1beta1ImportRagFilesRequest: A
      GoogleCloudAiplatformV1beta1ImportRagFilesRequest resource to be passed
      as the request body.
    parent: Required. The name of the RagCorpus resource into which to import
      files. Format:
      `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
  """
    googleCloudAiplatformV1beta1ImportRagFilesRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1ImportRagFilesRequest', 1)
    parent = _messages.StringField(2, required=True)