from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Automation(_messages.Message):
    """An `Automation` resource in the Cloud Deploy API. An `Automation`
  enables the automation of manually driven actions for a Delivery Pipeline,
  which includes Release promotion among Targets, Rollout repair and Rollout
  deployment strategy advancement. The intention of Automation is to reduce
  manual intervention in the continuous delivery process.

  Messages:
    AnnotationsValue: Optional. User annotations. These attributes can only be
      set and used by the user, and not by Cloud Deploy. Annotations must meet
      the following constraints: * Annotations are key/value pairs. * Valid
      annotation keys have two segments: an optional prefix and name,
      separated by a slash (`/`). * The name segment is required and must be
      63 characters or less, beginning and ending with an alphanumeric
      character (`[a-z0-9A-Z]`) with dashes (`-`), underscores (`_`), dots
      (`.`), and alphanumerics between. * The prefix is optional. If
      specified, the prefix must be a DNS subdomain: a series of DNS labels
      separated by dots(`.`), not longer than 253 characters in total,
      followed by a slash (`/`). See
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/annotations/#syntax-and-character-set for more details.
    LabelsValue: Optional. Labels are attributes that can be set and used by
      both the user and by Cloud Deploy. Labels must meet the following
      constraints: * Keys and values can contain only lowercase letters,
      numeric characters, underscores, and dashes. * All characters must use
      UTF-8 encoding, and international characters are allowed. * Keys must
      start with a lowercase letter or international character. * Each
      resource is limited to a maximum of 64 labels. Both keys and values are
      additionally constrained to be <= 63 characters.

  Fields:
    annotations: Optional. User annotations. These attributes can only be set
      and used by the user, and not by Cloud Deploy. Annotations must meet the
      following constraints: * Annotations are key/value pairs. * Valid
      annotation keys have two segments: an optional prefix and name,
      separated by a slash (`/`). * The name segment is required and must be
      63 characters or less, beginning and ending with an alphanumeric
      character (`[a-z0-9A-Z]`) with dashes (`-`), underscores (`_`), dots
      (`.`), and alphanumerics between. * The prefix is optional. If
      specified, the prefix must be a DNS subdomain: a series of DNS labels
      separated by dots(`.`), not longer than 253 characters in total,
      followed by a slash (`/`). See
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/annotations/#syntax-and-character-set for more details.
    createTime: Output only. Time at which the automation was created.
    description: Optional. Description of the `Automation`. Max length is 255
      characters.
    etag: Optional. The weak etag of the `Automation` resource. This checksum
      is computed by the server based on the value of other fields, and may be
      sent on update and delete requests to ensure the client has an up-to-
      date value before proceeding.
    labels: Optional. Labels are attributes that can be set and used by both
      the user and by Cloud Deploy. Labels must meet the following
      constraints: * Keys and values can contain only lowercase letters,
      numeric characters, underscores, and dashes. * All characters must use
      UTF-8 encoding, and international characters are allowed. * Keys must
      start with a lowercase letter or international character. * Each
      resource is limited to a maximum of 64 labels. Both keys and values are
      additionally constrained to be <= 63 characters.
    name: Output only. Name of the `Automation`. Format is `projects/{project}
      /locations/{location}/deliveryPipelines/{delivery_pipeline}/automations/
      {automation}`.
    rules: Required. List of Automation rules associated with the Automation
      resource. Must have at least one rule and limited to 250 rules per
      Delivery Pipeline. Note: the order of the rules here is not the same as
      the order of execution.
    selector: Required. Selected resources to which the automation will be
      applied.
    serviceAccount: Required. Email address of the user-managed IAM service
      account that creates Cloud Deploy release and rollout resources.
    suspended: Optional. When Suspended, automation is deactivated from
      execution.
    uid: Output only. Unique identifier of the `Automation`.
    updateTime: Output only. Time at which the automation was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. User annotations. These attributes can only be set and used
    by the user, and not by Cloud Deploy. Annotations must meet the following
    constraints: * Annotations are key/value pairs. * Valid annotation keys
    have two segments: an optional prefix and name, separated by a slash
    (`/`). * The name segment is required and must be 63 characters or less,
    beginning and ending with an alphanumeric character (`[a-z0-9A-Z]`) with
    dashes (`-`), underscores (`_`), dots (`.`), and alphanumerics between. *
    The prefix is optional. If specified, the prefix must be a DNS subdomain:
    a series of DNS labels separated by dots(`.`), not longer than 253
    characters in total, followed by a slash (`/`). See
    https://kubernetes.io/docs/concepts/overview/working-with-
    objects/annotations/#syntax-and-character-set for more details.

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
    63 characters.

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
    description = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    rules = _messages.MessageField('AutomationRule', 7, repeated=True)
    selector = _messages.MessageField('AutomationResourceSelector', 8)
    serviceAccount = _messages.StringField(9)
    suspended = _messages.BooleanField(10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)