from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesListRequest(_messages.Message):
    """A TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtFilesListRequest
  object.

  Fields:
    pageSize: Optional.
    pageToken: Optional. A token identifying a page of results the server
      should return. Typically, this is the value of
      ListAdaptiveMtFilesResponse.next_page_token returned from the previous
      call to `ListAdaptiveMtFiles` method. The first page is returned if
      `page_token`is empty or missing.
    parent: Required. The resource name of the project from which to list the
      Adaptive MT files.
      `projects/{project}/locations/{location}/adaptiveMtDatasets/{dataset}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)