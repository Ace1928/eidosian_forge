from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesListRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesListRequest object.

  Fields:
    pageSize: The maximum number of registries to return in the response. If
      this value is zero, the service will select a default size. A call may
      return fewer objects than requested. A non-empty `next_page_token` in
      the response indicates that more data is available.
    pageToken: The value returned by the last `ListDeviceRegistriesResponse`;
      indicates that this is a continuation of a prior `ListDeviceRegistries`
      call and the system should return the next page of data.
    parent: Required. The project and cloud region path. For example,
      `projects/example-project/locations/us-central1`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)