from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmActionsListRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmActionsListRequest object.

  Fields:
    pageSize: Maximum number of `WasmAction` resources to return per call. If
      not specified, at most 50 `WasmAction`s are returned. The maximum value
      is 1000; values above 1000 are coerced to 1000.
    pageToken: The value returned by the last `ListWasmActionsResponse` call.
      Indicates that this is a continuation of a prior `ListWasmActions` call,
      and that the next page of data is to be returned.
    parent: Required. The project and location from which the `WasmAction`
      resources are listed, specified in the following format:
      `projects/{project}/locations/global`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)