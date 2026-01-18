from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ProcessorVersion(_messages.Message):
    """A processor version is an implementation of a processor. Each processor
  can have multiple versions, pretrained by Google internally or uptrained by
  the customer. A processor can only have one default version at a time. Its
  document-processing behavior is defined by that version.

  Enums:
    ModelTypeValueValuesEnum: Output only. The model type of this processor
      version.
    StateValueValuesEnum: Output only. The state of the processor version.

  Fields:
    createTime: The time the processor version was created.
    deprecationInfo: If set, information about the eventual deprecation of
      this version.
    displayName: The display name of the processor version.
    documentSchema: The schema of the processor version. Describes the output.
    googleManaged: Output only. Denotes that this `ProcessorVersion` is
      managed by Google.
    kmsKeyName: The KMS key name used for encryption.
    kmsKeyVersionName: The KMS key version with which data is encrypted.
    latestEvaluation: The most recently invoked evaluation for the processor
      version.
    modelType: Output only. The model type of this processor version.
    name: Identifier. The resource name of the processor version. Format: `pro
      jects/{project}/locations/{location}/processors/{processor}/processorVer
      sions/{processor_version}`
    state: Output only. The state of the processor version.
  """

    class ModelTypeValueValuesEnum(_messages.Enum):
        """Output only. The model type of this processor version.

    Values:
      MODEL_TYPE_UNSPECIFIED: The processor version has unspecified model
        type.
      MODEL_TYPE_GENERATIVE: The processor version has generative model type.
      MODEL_TYPE_CUSTOM: The processor version has custom model type.
    """
        MODEL_TYPE_UNSPECIFIED = 0
        MODEL_TYPE_GENERATIVE = 1
        MODEL_TYPE_CUSTOM = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the processor version.

    Values:
      STATE_UNSPECIFIED: The processor version is in an unspecified state.
      DEPLOYED: The processor version is deployed and can be used for
        processing.
      DEPLOYING: The processor version is being deployed.
      UNDEPLOYED: The processor version is not deployed and cannot be used for
        processing.
      UNDEPLOYING: The processor version is being undeployed.
      CREATING: The processor version is being created.
      DELETING: The processor version is being deleted.
      FAILED: The processor version failed and is in an indeterminate state.
      IMPORTING: The processor version is being imported.
    """
        STATE_UNSPECIFIED = 0
        DEPLOYED = 1
        DEPLOYING = 2
        UNDEPLOYED = 3
        UNDEPLOYING = 4
        CREATING = 5
        DELETING = 6
        FAILED = 7
        IMPORTING = 8
    createTime = _messages.StringField(1)
    deprecationInfo = _messages.MessageField('GoogleCloudDocumentaiV1ProcessorVersionDeprecationInfo', 2)
    displayName = _messages.StringField(3)
    documentSchema = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchema', 4)
    googleManaged = _messages.BooleanField(5)
    kmsKeyName = _messages.StringField(6)
    kmsKeyVersionName = _messages.StringField(7)
    latestEvaluation = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationReference', 8)
    modelType = _messages.EnumField('ModelTypeValueValuesEnum', 9)
    name = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)