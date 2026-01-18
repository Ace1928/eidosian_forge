from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaDeleteRequest object.

  Fields:
    force: Optional. If set to true, any RagFiles in this RagCorpus will also
      be deleted. Otherwise, the request will only work if the RagCorpus has
      no RagFiles.
    name: Required. The name of the RagCorpus resource to be deleted. Format:
      `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)