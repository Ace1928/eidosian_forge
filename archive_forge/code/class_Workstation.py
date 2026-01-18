from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Workstation(_messages.Message):
    """A single instance of a developer workstation with its own persistent
  storage.

  Enums:
    StateValueValuesEnum: Output only. Current state of the workstation.

  Messages:
    AnnotationsValue: Optional. Client-specified annotations.
    EnvValue: Optional. Environment variables passed to the workstation
      container's entrypoint.
    LabelsValue: Optional.
      [Labels](https://cloud.google.com/workstations/docs/label-resources)
      that are applied to the workstation and that are also propagated to the
      underlying Compute Engine resources.

  Fields:
    annotations: Optional. Client-specified annotations.
    createTime: Output only. Time when this workstation was created.
    deleteTime: Output only. Time when this workstation was soft-deleted.
    displayName: Optional. Human-readable name for this workstation.
    env: Optional. Environment variables passed to the workstation container's
      entrypoint.
    etag: Optional. Checksum computed by the server. May be sent on update and
      delete requests to make sure that the client has an up-to-date value
      before proceeding.
    host: Output only. Host to which clients can send HTTPS traffic that will
      be received by the workstation. Authorized traffic will be received to
      the workstation as HTTP on port 80. To send traffic to a different port,
      clients may prefix the host with the destination port in the format
      `{port}-{host}`.
    kmsKey: Output only. The name of the Google Cloud KMS encryption key used
      to encrypt this workstation. The KMS key can only be configured in the
      WorkstationConfig. The expected format is
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
    labels: Optional.
      [Labels](https://cloud.google.com/workstations/docs/label-resources)
      that are applied to the workstation and that are also propagated to the
      underlying Compute Engine resources.
    name: Identifier. Full name of this workstation.
    reconciling: Output only. Indicates whether this workstation is currently
      being updated to match its intended state.
    startTime: Output only. Time when this workstation was most recently
      successfully started, regardless of the workstation's initial state.
    state: Output only. Current state of the workstation.
    uid: Output only. A system-assigned unique identifier for this
      workstation.
    updateTime: Output only. Time when this workstation was most recently
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the workstation.

    Values:
      STATE_UNSPECIFIED: Do not use.
      STATE_STARTING: The workstation is not yet ready to accept requests from
        users but will be soon.
      STATE_RUNNING: The workstation is ready to accept requests from users.
      STATE_STOPPING: The workstation is being stopped.
      STATE_STOPPED: The workstation is stopped and will not be able to
        receive requests until it is started.
    """
        STATE_UNSPECIFIED = 0
        STATE_STARTING = 1
        STATE_RUNNING = 2
        STATE_STOPPING = 3
        STATE_STOPPED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Client-specified annotations.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvValue(_messages.Message):
        """Optional. Environment variables passed to the workstation container's
    entrypoint.

    Messages:
      AdditionalProperty: An additional property for a EnvValue object.

    Fields:
      additionalProperties: Additional properties of type EnvValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. [Labels](https://cloud.google.com/workstations/docs/label-
    resources) that are applied to the workstation and that are also
    propagated to the underlying Compute Engine resources.

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    createTime = _messages.StringField(2)
    deleteTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    env = _messages.MessageField('EnvValue', 5)
    etag = _messages.StringField(6)
    host = _messages.StringField(7)
    kmsKey = _messages.StringField(8)
    labels = _messages.MessageField('LabelsValue', 9)
    name = _messages.StringField(10)
    reconciling = _messages.BooleanField(11)
    startTime = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    uid = _messages.StringField(14)
    updateTime = _messages.StringField(15)