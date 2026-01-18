from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsServerTlsPoliciesCreateRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsServerTlsPoliciesCreateRequest object.

  Fields:
    parent: Required. The parent resource of the ServerTlsPolicy. Must be in
      the format `projects/*/locations/{location}`.
    serverTlsPolicy: A ServerTlsPolicy resource to be passed as the request
      body.
    serverTlsPolicyId: Required. Short name of the ServerTlsPolicy resource to
      be created. This value should be 1-63 characters long, containing only
      letters, numbers, hyphens, and underscores, and should not start with a
      number. E.g. "server_mtls_policy".
  """
    parent = _messages.StringField(1, required=True)
    serverTlsPolicy = _messages.MessageField('ServerTlsPolicy', 2)
    serverTlsPolicyId = _messages.StringField(3)