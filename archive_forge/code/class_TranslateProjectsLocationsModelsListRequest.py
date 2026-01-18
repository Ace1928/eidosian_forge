from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsModelsListRequest(_messages.Message):
    """A TranslateProjectsLocationsModelsListRequest object.

  Fields:
    filter: Optional. An expression for filtering the models that will be
      returned. Supported filter: `dataset_id=${dataset_id}`
    pageSize: Optional. Requested page size. The server can return fewer
      results than requested.
    pageToken: Optional. A token identifying a page of results for the server
      to return. Typically obtained from next_page_token field in the response
      of a ListModels call.
    parent: Required. Name of the parent project. In form of
      `projects/{project-number-or-id}/locations/{location-id}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)