from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WasmPluginVersionDetails(_messages.Message):
    """Details of a `WasmPluginVersion` resource to be inlined in the
  `WasmPlugin` resource.

  Messages:
    LabelsValue: Optional. Set of labels associated with the
      `WasmPluginVersion` resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A human-readable description of the resource.
    imageDigest: Output only. The resolved digest for the image specified in
      `image`. The digest is resolved during the creation of a
      `WasmPluginVersion` resource. This field holds the digest value
      regardless of whether a tag or digest was originally specified in the
      `image` field.
    imageUri: Required. URI of the container image containing the Wasm module,
      stored in the Artifact Registry. The container image must contain only a
      single file with the name `plugin.wasm`. When a new `WasmPluginVersion`
      resource is created, the URI gets resolved to an image digest and saved
      in the `image_digest` field.
    labels: Optional. Set of labels associated with the `WasmPluginVersion`
      resource.
    pluginConfig: Optional. Configuration for the Wasm plugin. The
      configuration is provided to the Proxy-Wasm plugin at runtime by using
      the `ON_CONFIGURE` callback. Deprecated: Use `plugin_config_data` or
      `plugin_config_uri`.
    pluginConfigData: Configuration for the Wasm plugin. The configuration is
      provided to the Wasm plugin at runtime through the `ON_CONFIGURE`
      callback. When a new `WasmPluginVersion` version is created, the digest
      of the contents is saved in the `plugin_config_digest` field.
    pluginConfigDigest: Output only. This field holds the digest (usually
      checksum) value for the plugin configuration. The value is calculated
      based on the contents of the `plugin_config_data` or the container image
      defined by the `plugin_config_uri` field.
    pluginConfigUri: URI of the WasmPlugin configuration stored in the
      Artifact Registry. The configuration is provided to the Wasm plugin at
      runtime through the `ON_CONFIGURE` callback. The container image must
      contain only a single file with the name `plugin.config`. When a new
      `WasmPluginVersion` resource is created, the digest of the container
      image is saved in the `plugin_config_digest` field.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of labels associated with the `WasmPluginVersion`
    resource.

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
    imageDigest = _messages.StringField(3)
    imageUri = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    pluginConfig = _messages.BytesField(6)
    pluginConfigData = _messages.BytesField(7)
    pluginConfigDigest = _messages.StringField(8)
    pluginConfigUri = _messages.StringField(9)
    updateTime = _messages.StringField(10)