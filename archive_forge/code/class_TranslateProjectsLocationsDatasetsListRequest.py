from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsDatasetsListRequest(_messages.Message):
    """A TranslateProjectsLocationsDatasetsListRequest object.

  Fields:
    pageSize: Optional. Requested page size. The server can return fewer
      results than requested.
    pageToken: Optional. A token identifying a page of results for the server
      to return. Typically obtained from next_page_token field in the response
      of a ListDatasets call.
    parent: Required. Name of the parent project. In form of
      `projects/{project-number-or-id}/locations/{location-id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)