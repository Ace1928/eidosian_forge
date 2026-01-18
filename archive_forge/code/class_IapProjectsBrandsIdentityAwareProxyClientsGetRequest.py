from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsIdentityAwareProxyClientsGetRequest(_messages.Message):
    """A IapProjectsBrandsIdentityAwareProxyClientsGetRequest object.

  Fields:
    name: Required. Name of the Identity Aware Proxy client to be fetched. In
      the following format: projects/{project_number/id}/brands/{brand}/identi
      tyAwareProxyClients/{client_id}.
  """
    name = _messages.StringField(1, required=True)