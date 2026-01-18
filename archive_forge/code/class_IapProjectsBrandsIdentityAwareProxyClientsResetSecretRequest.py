from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsIdentityAwareProxyClientsResetSecretRequest(_messages.Message):
    """A IapProjectsBrandsIdentityAwareProxyClientsResetSecretRequest object.

  Fields:
    name: Required. Name of the Identity Aware Proxy client to that will have
      its secret reset. In the following format: projects/{project_number/id}/
      brands/{brand}/identityAwareProxyClients/{client_id}.
    resetIdentityAwareProxyClientSecretRequest: A
      ResetIdentityAwareProxyClientSecretRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    resetIdentityAwareProxyClientSecretRequest = _messages.MessageField('ResetIdentityAwareProxyClientSecretRequest', 2)