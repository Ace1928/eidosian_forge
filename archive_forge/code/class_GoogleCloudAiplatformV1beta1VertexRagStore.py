from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1VertexRagStore(_messages.Message):
    """Retrieve from Vertex RAG Store for grounding.

  Fields:
    ragCorpora: Required. Vertex RAG Store corpus resource name:
      `projects/{project}/locations/{location}/ragCorpora/{ragCorpus}`
      Currently only one corpus is allowed. In the future we may open up
      multiple corpora support. However, they should be from the same project
      and location.
    similarityTopK: Optional. Number of top k results to return from the
      selected corpora.
  """
    ragCorpora = _messages.StringField(1, repeated=True)
    similarityTopK = _messages.IntegerField(2, variant=_messages.Variant.INT32)