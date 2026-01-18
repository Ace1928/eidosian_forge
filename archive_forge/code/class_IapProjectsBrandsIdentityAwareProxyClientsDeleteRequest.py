from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsIdentityAwareProxyClientsDeleteRequest(_messages.Message):
    """A IapProjectsBrandsIdentityAwareProxyClientsDeleteRequest object.

  Fields:
    name: Required. Name of the Identity Aware Proxy client to be deleted. In
      the following format: projects/{project_number/id}/brands/{brand}/identi
      tyAwareProxyClients/{client_id}.
  """
    name = _messages.StringField(1, required=True)