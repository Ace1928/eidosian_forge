from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsDatasetsExamplesListRequest(_messages.Message):
    """A TranslateProjectsLocationsDatasetsExamplesListRequest object.

  Fields:
    filter: Optional. An expression for filtering the examples that will be
      returned. Example filter: * `usage=TRAIN`
    pageSize: Optional. Requested page size. The server can return fewer
      results than requested.
    pageToken: Optional. A token identifying a page of results for the server
      to return. Typically obtained from next_page_token field in the response
      of a ListExamples call.
    parent: Required. Name of the parent dataset. In form of
      `projects/{project-number-or-id}/locations/{location-
      id}/datasets/{dataset-id}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)