from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeliveryPipeline(_messages.Message):
    """A `DeliveryPipeline` resource in the Cloud Deploy API. A
  `DeliveryPipeline` defines a pipeline through which a Skaffold configuration
  can progress.

  Messages:
    AnnotationsValue: User annotations. These attributes can only be set and
      used by the user, and not by Cloud Deploy.
    LabelsValue: Labels are attributes that can be set and used by both the
      user and by Cloud Deploy. Labels must meet the following constraints: *
      Keys and values can contain only lowercase letters, numeric characters,
      underscores, and dashes. * All characters must use UTF-8 encoding, and
      international characters are allowed. * Keys must start with a lowercase
      letter or international character. * Each resource is limited to a
      maximum of 64 labels. Both keys and values are additionally constrained
      to be <= 128 bytes.

  Fields:
    annotations: User annotations. These attributes can only be set and used
      by the user, and not by Cloud Deploy.
    condition: Output only. Information around the state of the Delivery
      Pipeline.
    createTime: Output only. Time at which the pipeline was created.
    description: Description of the `DeliveryPipeline`. Max length is 255
      characters.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding.
    labels: Labels are attributes that can be set and used by both the user
      and by Cloud Deploy. Labels must meet the following constraints: * Keys
      and values can contain only lowercase letters, numeric characters,
      underscores, and dashes. * All characters must use UTF-8 encoding, and
      international characters are allowed. * Keys must start with a lowercase
      letter or international character. * Each resource is limited to a
      maximum of 64 labels. Both keys and values are additionally constrained
      to be <= 128 bytes.
    name: Optional. Name of the `DeliveryPipeline`. Format is
      `projects/{project}/locations/{location}/deliveryPipelines/a-z{0,62}`.
    serialPipeline: SerialPipeline defines a sequential set of stages for a
      `DeliveryPipeline`.
    suspended: When suspended, no new releases or rollouts can be created, but
      in-progress ones will complete.
    uid: Output only. Unique identifier of the `DeliveryPipeline`.
    updateTime: Output only. Most recent time at which the pipeline was
      updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """User annotations. These attributes can only be set and used by the
    user, and not by Cloud Deploy.

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
    class LabelsValue(_messages.Message):
        """Labels are attributes that can be set and used by both the user and by
    Cloud Deploy. Labels must meet the following constraints: * Keys and
    values can contain only lowercase letters, numeric characters,
    underscores, and dashes. * All characters must use UTF-8 encoding, and
    international characters are allowed. * Keys must start with a lowercase
    letter or international character. * Each resource is limited to a maximum
    of 64 labels. Both keys and values are additionally constrained to be <=
    128 bytes.

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
    condition = _messages.MessageField('PipelineCondition', 2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    etag = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    serialPipeline = _messages.MessageField('SerialPipeline', 8)
    suspended = _messages.BooleanField(9)
    uid = _messages.StringField(10)
    updateTime = _messages.StringField(11)