from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsListRequest(_messages.Message):
    """A TranslateProjectsLocationsAdaptiveMtDatasetsListRequest object.

  Fields:
    filter: Optional. An expression for filtering the results of the request.
      Filter is not supported yet.
    pageSize: Optional. Requested page size. The server may return fewer
      results than requested. If unspecified, the server picks an appropriate
      default.
    pageToken: Optional. A token identifying a page of results the server
      should return. Typically, this is the value of
      ListAdaptiveMtDatasetsResponse.next_page_token returned from the
      previous call to `ListAdaptiveMtDatasets` method. The first page is
      returned if `page_token`is empty or missing.
    parent: Required. The resource name of the project from which to list the
      Adaptive MT datasets. `projects/{project-number-or-
      id}/locations/{location-id}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)