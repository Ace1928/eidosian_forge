from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPeeringsCreateRequest(_messages.Message):
    """A VmwareengineProjectsLocationsNetworkPeeringsCreateRequest object.

  Fields:
    networkPeering: A NetworkPeering resource to be passed as the request
      body.
    networkPeeringId: Required. The user-provided identifier of the new
      `NetworkPeering`. This identifier must be unique among `NetworkPeering`
      resources within the parent and becomes the final token in the name URI.
      The identifier must meet the following requirements: * Only contains
      1-63 alphanumeric characters and hyphens * Begins with an alphabetical
      character * Ends with a non-hyphen character * Not formatted as a UUID *
      Complies with [RFC 1034](https://datatracker.ietf.org/doc/html/rfc1034)
      (section 3.5)
    parent: Required. The resource name of the location to create the new
      network peering in. This value is always `global`, because
      `NetworkPeering` is a global resource. Resource names are schemeless
      URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/global`
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check if
      original operation with the same request ID was received, and if so,
      will ignore the second request. This prevents clients from accidentally
      creating duplicate commitments. The request ID must be a valid UUID with
      the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    networkPeering = _messages.MessageField('NetworkPeering', 1)
    networkPeeringId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)