from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsAuthorizationPoliciesListRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsAuthorizationPoliciesListRequest
  object.

  Fields:
    pageSize: Maximum number of AuthorizationPolicies to return per call.
    pageToken: The value returned by the last
      `ListAuthorizationPoliciesResponse` Indicates that this is a
      continuation of a prior `ListAuthorizationPolicies` call, and that the
      system should return the next page of data.
    parent: Required. The project and location from which the
      AuthorizationPolicies should be listed, specified in the format
      `projects/{project}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)