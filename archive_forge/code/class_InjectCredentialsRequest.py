from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InjectCredentialsRequest(_messages.Message):
    """A request to inject credentials into a cluster.

  Fields:
    clusterUuid: Required. The cluster UUID.
    credentialsCiphertext: Required. The encrypted credentials being injected
      in to the cluster.The client is responsible for encrypting the
      credentials in a way that is supported by the cluster.A wrapped value is
      used here so that the actual contents of the encrypted credentials are
      not written to audit logs.
  """
    clusterUuid = _messages.StringField(1)
    credentialsCiphertext = _messages.StringField(2)