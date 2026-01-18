from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomTargetType(_messages.Message):
    """A `CustomTargetType` resource in the Cloud Deploy API. A
  `CustomTargetType` defines a type of custom target that can be referenced in
  a `Target` in order to facilitate deploying to other systems besides the
  supported runtimes.

  Messages:
    AnnotationsValue: Optional. User annotations. These attributes can only be
      set and used by the user, and not by Cloud Deploy. See
      https://google.aip.dev/128#annotations for more details such as format
      and size limitations.
    LabelsValue: Optional. Labels are attributes that can be set and used by
      both the user and by Cloud Deploy. Labels must meet the following
      constraints: * Keys and values can contain only lowercase letters,
      numeric characters, underscores, and dashes. * All characters must use
      UTF-8 encoding, and international characters are allowed. * Keys must
      start with a lowercase letter or international character. * Each
      resource is limited to a maximum of 64 labels. Both keys and values are
      additionally constrained to be <= 128 bytes.

  Fields:
    annotations: Optional. User annotations. These attributes can only be set
      and used by the user, and not by Cloud Deploy. See
      https://google.aip.dev/128#annotations for more details such as format
      and size limitations.
    createTime: Output only. Time at which the `CustomTargetType` was created.
    customActions: Configures render and deploy for the `CustomTargetType`
      using Skaffold custom actions.
    customTargetTypeId: Output only. Resource id of the `CustomTargetType`.
    description: Optional. Description of the `CustomTargetType`. Max length
      is 255 characters.
    etag: Optional. This checksum is computed by the server based on the value
      of other fields, and may be sent on update and delete requests to ensure
      the client has an up-to-date value before proceeding.
    labels: Optional. Labels are attributes that can be set and used by both
      the user and by Cloud Deploy. Labels must meet the following
      constraints: * Keys and values can contain only lowercase letters,
      numeric characters, underscores, and dashes. * All characters must use
      UTF-8 encoding, and international characters are allowed. * Keys must
      start with a lowercase letter or international character. * Each
      resource is limited to a maximum of 64 labels. Both keys and values are
      additionally constrained to be <= 128 bytes.
    name: Optional. Name of the `CustomTargetType`. Format is
      `projects/{project}/locations/{location}/customTargetTypes/a-z{0,62}`.
    uid: Output only. Unique identifier of the `CustomTargetType`.
    updateTime: Output only. Most recent time at which the `CustomTargetType`
      was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. User annotations. These attributes can only be set and used
    by the user, and not by Cloud Deploy. See
    https://google.aip.dev/128#annotations for more details such as format and
    size limitations.

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
        """Optional. Labels are attributes that can be set and used by both the
    user and by Cloud Deploy. Labels must meet the following constraints: *
    Keys and values can contain only lowercase letters, numeric characters,
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
    createTime = _messages.StringField(2)
    customActions = _messages.MessageField('CustomTargetSkaffoldActions', 3)
    customTargetTypeId = _messages.StringField(4)
    description = _messages.StringField(5)
    etag = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)