from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SavedAddress(_messages.Message):
    """An address where appliances can be shipped.

  Messages:
    AnnotationsValue: User annotations. See
      https://google.aip.dev/128#annotations.
    LabelsValue: Labels as key value pairs. Labels must meet the following
      constraints: * Keys and values can contain only lowercase letters,
      numeric characters, underscores, and dashes. * All characters must use
      UTF-8 encoding, and international characters are allowed. * Keys must
      start with a lowercase letter or international character. * Each
      resource is limited to a maximum of 64 labels. * Label keys must be
      between 1 and 63 characters long and must conform to the following
      regular expression: a-z{0,62}. * Label values must be between 0 and 63
      characters long and must conform to the regular expression
      [a-z0-9_-]{0,63}.

  Fields:
    address: The saved shipping address.
    annotations: User annotations. See https://google.aip.dev/128#annotations.
    createTime: Output only. Create time.
    deleteTime: Output only. Delete time stamp.
    displayName: A mutable, user-settable name for the resource. It does not
      need to be unique and should be less than 64 characters.
    etag: Strongly validated etag, computed by the server based on the value
      of other fields, and may be sent on update and delete requests to ensure
      the client has an up-to-date value before proceeding. See
      https://google.aip.dev/154.
    labels: Labels as key value pairs. Labels must meet the following
      constraints: * Keys and values can contain only lowercase letters,
      numeric characters, underscores, and dashes. * All characters must use
      UTF-8 encoding, and international characters are allowed. * Keys must
      start with a lowercase letter or international character. * Each
      resource is limited to a maximum of 64 labels. * Label keys must be
      between 1 and 63 characters long and must conform to the following
      regular expression: a-z{0,62}. * Label values must be between 0 and 63
      characters long and must conform to the regular expression
      [a-z0-9_-]{0,63}.
    name: name of resource.
    reconciling: Output only. Reconciling
      (https://google.aip.dev/128#reconciliation). Set to true if the current
      state of SavedAddress does not match the user's intended state, and the
      service is actively updating the resource to reconcile them. This can
      happen due to user-triggered updates or system actions like failover or
      maintenance.
    shippingContact: Contact information used for the shipments to the given
      address.
    uid: Output only. A system-assigned, unique identifier (UUID4) for the
      resource.
    updateTime: Output only. Update time.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """User annotations. See https://google.aip.dev/128#annotations.

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
        """Labels as key value pairs. Labels must meet the following constraints:
    * Keys and values can contain only lowercase letters, numeric characters,
    underscores, and dashes. * All characters must use UTF-8 encoding, and
    international characters are allowed. * Keys must start with a lowercase
    letter or international character. * Each resource is limited to a maximum
    of 64 labels. * Label keys must be between 1 and 63 characters long and
    must conform to the following regular expression: a-z{0,62}. * Label
    values must be between 0 and 63 characters long and must conform to the
    regular expression [a-z0-9_-]{0,63}.

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
    address = _messages.MessageField('PostalAddress', 1)
    annotations = _messages.MessageField('AnnotationsValue', 2)
    createTime = _messages.StringField(3)
    deleteTime = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    reconciling = _messages.BooleanField(9)
    shippingContact = _messages.MessageField('ContactInfo', 10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)