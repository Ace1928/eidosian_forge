from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Extension(_messages.Message):
    """Extensions are tools for large language models to access external data,
  run computations, etc.

  Fields:
    createTime: Output only. Timestamp when this Extension was created.
    description: Optional. The description of the Extension.
    displayName: Required. The display name of the Extension. The name can be
      up to 128 characters long and can consist of any UTF-8 characters.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    extensionOperations: Output only. Supported operations.
    manifest: Required. Manifest of the Extension.
    name: Identifier. The resource name of the Extension.
    privateServiceConnectConfig: Optional. The PrivateServiceConnect config
      for the extension. If specified, the service endpoints associated with
      the Extension should be registered with private network access in the
      provided Service Directory (https://cloud.google.com/service-
      directory/docs/configuring-private-network-access). If the service
      contains more than one endpoint with a network, the service will
      arbitrarilty choose one of the endpoints to use for extension execution.
    runtimeConfig: Optional. Runtime config controlling the runtime behavior
      of this Extension.
    toolUseExamples: Optional. Examples to illustrate the usage of the
      extension as a tool.
    updateTime: Output only. Timestamp when this Extension was most recently
      updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    extensionOperations = _messages.MessageField('GoogleCloudAiplatformV1beta1ExtensionOperation', 5, repeated=True)
    manifest = _messages.MessageField('GoogleCloudAiplatformV1beta1ExtensionManifest', 6)
    name = _messages.StringField(7)
    privateServiceConnectConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1ExtensionPrivateServiceConnectConfig', 8)
    runtimeConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1RuntimeConfig', 9)
    toolUseExamples = _messages.MessageField('GoogleCloudAiplatformV1beta1ToolUseExample', 10, repeated=True)
    updateTime = _messages.StringField(11)