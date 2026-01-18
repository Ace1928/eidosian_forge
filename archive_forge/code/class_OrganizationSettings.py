from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrganizationSettings(_messages.Message):
    """User specified settings that are attached to the Security Command Center
  organization.

  Fields:
    assetDiscoveryConfig: The configuration used for Asset Discovery runs.
    enableAssetDiscovery: A flag that indicates if Asset Discovery should be
      enabled. If the flag is set to `true`, then discovery of assets will
      occur. If it is set to `false`, all historical assets will remain, but
      discovery of future assets will not occur.
    name: The relative resource name of the settings. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me Example: "organizations/{organization_id}/organizationSettings".
  """
    assetDiscoveryConfig = _messages.MessageField('AssetDiscoveryConfig', 1)
    enableAssetDiscovery = _messages.BooleanField(2)
    name = _messages.StringField(3)