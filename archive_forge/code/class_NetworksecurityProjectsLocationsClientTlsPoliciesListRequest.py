from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsClientTlsPoliciesListRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsClientTlsPoliciesListRequest object.

  Fields:
    pageSize: Maximum number of ClientTlsPolicies to return per call.
    pageToken: The value returned by the last `ListClientTlsPoliciesResponse`
      Indicates that this is a continuation of a prior `ListClientTlsPolicies`
      call, and that the system should return the next page of data.
    parent: Required. The project and location from which the
      ClientTlsPolicies should be listed, specified in the format
      `projects/*/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)