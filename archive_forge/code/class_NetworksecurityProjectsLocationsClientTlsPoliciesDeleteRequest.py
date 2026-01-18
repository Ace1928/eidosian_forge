from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsClientTlsPoliciesDeleteRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsClientTlsPoliciesDeleteRequest object.

  Fields:
    name: Required. A name of the ClientTlsPolicy to delete. Must be in the
      format `projects/*/locations/{location}/clientTlsPolicies/*`.
  """
    name = _messages.StringField(1, required=True)