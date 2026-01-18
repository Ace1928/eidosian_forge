from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaRagFilesGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaRagFilesGetRequest object.

  Fields:
    name: Required. The name of the RagFile resource. Format: `projects/{proje
      ct}/locations/{location}/ragCorpora/{rag_corpus}/ragFiles/{rag_file}`
  """
    name = _messages.StringField(1, required=True)