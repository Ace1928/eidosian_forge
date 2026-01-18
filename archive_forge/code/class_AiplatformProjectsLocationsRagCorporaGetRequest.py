from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaGetRequest object.

  Fields:
    name: Required. The name of the RagCorpus resource. Format:
      `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
  """
    name = _messages.StringField(1, required=True)