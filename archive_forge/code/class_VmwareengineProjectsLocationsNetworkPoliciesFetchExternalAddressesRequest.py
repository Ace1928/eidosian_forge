from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPoliciesFetchExternalAddressesRequest(_messages.Message):
    """A
  VmwareengineProjectsLocationsNetworkPoliciesFetchExternalAddressesRequest
  object.

  Fields:
    networkPolicy: Required. The resource name of the network policy to query
      for assigned external IP addresses. Resource names are schemeless URIs
      that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/networkPolicies/my-policy`
    pageSize: The maximum number of external IP addresses to return in one
      page. The service may return fewer than this value. The maximum value is
      coerced to 1000. The default value of this field is 500.
    pageToken: A page token, received from a previous
      `FetchNetworkPolicyExternalAddresses` call. Provide this to retrieve the
      subsequent page. When paginating, all parameters provided to
      `FetchNetworkPolicyExternalAddresses`, except for `page_size` and
      `page_token`, must match the call that provided the page token.
  """
    networkPolicy = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)