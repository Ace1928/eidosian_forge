from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MetadataStore(_messages.Message):
    """Instance of a metadata store. Contains a set of metadata that can be
  queried.

  Fields:
    createTime: Output only. Timestamp when this MetadataStore was created.
    description: Description of the MetadataStore.
    encryptionSpec: Customer-managed encryption key spec for a Metadata Store.
      If set, this Metadata Store and all sub-resources of this Metadata Store
      are secured using this key.
    name: Output only. The resource name of the MetadataStore instance.
    state: Output only. State information of the MetadataStore.
    updateTime: Output only. Timestamp when this MetadataStore was last
      updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 3)
    name = _messages.StringField(4)
    state = _messages.MessageField('GoogleCloudAiplatformV1beta1MetadataStoreMetadataStoreState', 5)
    updateTime = _messages.StringField(6)