from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsServerTlsPoliciesGetRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsServerTlsPoliciesGetRequest object.

  Fields:
    name: Required. A name of the ServerTlsPolicy to get. Must be in the
      format `projects/*/locations/{location}/serverTlsPolicies/*`.
  """
    name = _messages.StringField(1, required=True)