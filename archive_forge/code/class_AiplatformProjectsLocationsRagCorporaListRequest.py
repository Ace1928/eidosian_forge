from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsRagCorporaListRequest(_messages.Message):
    """A AiplatformProjectsLocationsRagCorporaListRequest object.

  Fields:
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token. Typically obtained via
      ListRagCorporaResponse.next_page_token of the previous
      VertexRagDataService.ListRagCorpora call.
    parent: Required. The resource name of the Location from which to list the
      RagCorpora. Format: `projects/{project}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)