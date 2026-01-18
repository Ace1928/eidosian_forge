from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Asset(_messages.Message):
    """An asset represents a cloud resource that is being managed within a lake
  as a member of a zone.

  Enums:
    StateValueValuesEnum: Output only. Current state of the asset.

  Messages:
    LabelsValue: Optional. User defined labels for the asset.

  Fields:
    createTime: Output only. The time when the asset was created.
    description: Optional. Description of the asset.
    discoverySpec: Optional. Specification of the discovery feature applied to
      data referenced by this asset. When this spec is left unset, the asset
      will use the spec set on the parent zone.
    discoveryStatus: Output only. Status of the discovery feature applied to
      data referenced by this asset.
    displayName: Optional. User friendly display name.
    labels: Optional. User defined labels for the asset.
    name: Output only. The relative resource name of the asset, of the form: p
      rojects/{project_number}/locations/{location_id}/lakes/{lake_id}/zones/{
      zone_id}/assets/{asset_id}.
    resourceSpec: Required. Specification of the resource that is referenced
      by this asset.
    resourceStatus: Output only. Status of the resource referenced by this
      asset.
    securityStatus: Output only. Status of the security policy applied to
      resource referenced by this asset.
    state: Output only. Current state of the asset.
    uid: Output only. System generated globally unique ID for the asset. This
      ID will be different if the asset is deleted and re-created with the
      same name.
    updateTime: Output only. The time when the asset was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the asset.

    Values:
      STATE_UNSPECIFIED: State is not specified.
      ACTIVE: Resource is active, i.e., ready to use.
      CREATING: Resource is under creation.
      DELETING: Resource is under deletion.
      ACTION_REQUIRED: Resource is active but has unresolved actions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3
        ACTION_REQUIRED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User defined labels for the asset.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    discoverySpec = _messages.MessageField('GoogleCloudDataplexV1AssetDiscoverySpec', 3)
    discoveryStatus = _messages.MessageField('GoogleCloudDataplexV1AssetDiscoveryStatus', 4)
    displayName = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    resourceSpec = _messages.MessageField('GoogleCloudDataplexV1AssetResourceSpec', 8)
    resourceStatus = _messages.MessageField('GoogleCloudDataplexV1AssetResourceStatus', 9)
    securityStatus = _messages.MessageField('GoogleCloudDataplexV1AssetSecurityStatus', 10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    uid = _messages.StringField(12)
    updateTime = _messages.StringField(13)