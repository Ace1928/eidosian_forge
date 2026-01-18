from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectionsListRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectionsListRequest object.

  Fields:
    filter: Optional. A filter specifying constraints of a list operation.
    orderBy: Optional. Specifies the ordering of results. See [Sorting
      order](https://cloud.google.com/apis/design/design_patterns#sorting_orde
      r) for more information.
    pageSize: Optional. The maximum number of items to return. If not
      specified, a default value of 50 will be used by the service. Regardless
      of the page_size value, the response may include a partial list and a
      caller should only rely on response's next_page_token to determine if
      there are more instances left to be queried.
    pageToken: Optional. The next_page_token value returned from a previous
      ListAppConnectionsRequest, if any.
    parent: Required. The resource name of the AppConnection location using
      the form: `projects/{project_id}/locations/{location_id}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)