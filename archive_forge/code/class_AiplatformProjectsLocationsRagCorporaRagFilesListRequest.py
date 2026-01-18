from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaRagFilesListRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaRagFilesListRequest object.

  Fields:
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListRagFilesResponse.next_page_token of the previous
      VertexRagDataService.ListRagFiles call.
    parent: Required. The resource name of the RagCorpus from which to list
      the RagFiles. Format:
      `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)