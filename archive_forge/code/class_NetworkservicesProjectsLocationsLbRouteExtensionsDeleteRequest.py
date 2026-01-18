from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsLbRouteExtensionsDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsLbRouteExtensionsDeleteRequest object.

  Fields:
    name: Required. The name of the `LbRouteExtension` resource to delete.
      Must be in the format `projects/{project}/locations/{location}/lbRouteEx
      tensions/{lb_route_extension}`.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      can ignore the request if it has already been completed. The server
      guarantees that for at least 60 minutes after the first request. For
      example, consider a situation where you make an initial request and the
      request times out. If you make the request again with the same request
      ID, the server can check if original operation with the same request ID
      was received, and if so, ignores the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    name = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)