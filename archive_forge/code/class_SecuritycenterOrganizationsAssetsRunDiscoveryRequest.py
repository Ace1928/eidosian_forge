from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsAssetsRunDiscoveryRequest(_messages.Message):
    """A SecuritycenterOrganizationsAssetsRunDiscoveryRequest object.

  Fields:
    parent: Required. Name of the organization to run asset discovery for. Its
      format is "organizations/[organization_id]".
    runAssetDiscoveryRequest: A RunAssetDiscoveryRequest resource to be passed
      as the request body.
  """
    parent = _messages.StringField(1, required=True)
    runAssetDiscoveryRequest = _messages.MessageField('RunAssetDiscoveryRequest', 2)