from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Trigger(_messages.Message):
    """A representation of the trigger resource.

  Messages:
    ConditionsValue: Output only. The reason(s) why a trigger is in FAILED
      state.
    LabelsValue: Optional. User labels attached to the triggers that can be
      used to group resources.

  Fields:
    channel: Optional. The name of the channel associated with the trigger in
      `projects/{project}/locations/{location}/channels/{channel}` format. You
      must provide a channel to receive events from Eventarc SaaS partners.
    conditions: Output only. The reason(s) why a trigger is in FAILED state.
    createTime: Output only. The creation time.
    destination: Required. Destination specifies where the events should be
      sent to.
    etag: Output only. This checksum is computed by the server based on the
      value of other fields, and might be sent only on create requests to
      ensure that the client has an up-to-date value before proceeding.
    eventDataContentType: Optional. EventDataContentType specifies the type of
      payload in MIME format that is expected from the CloudEvent data field.
      This is set to `application/json` if the value is not defined.
    eventFilters: Required. Unordered list. The list of filters that applies
      to event attributes. Only events that match all the provided filters are
      sent to the destination.
    labels: Optional. User labels attached to the triggers that can be used to
      group resources.
    name: Required. The resource name of the trigger. Must be unique within
      the location of the project and must be in
      `projects/{project}/locations/{location}/triggers/{trigger}` format.
    serviceAccount: Optional. The IAM service account email associated with
      the trigger. The service account represents the identity of the trigger.
      The `iam.serviceAccounts.actAs` permission must be granted on the
      service account to allow a principal to impersonate the service account.
      For more information, see the [Roles and
      permissions](/eventarc/docs/all-roles-permissions) page specific to the
      trigger destination.
    transport: Optional. To deliver messages, Eventarc might use other Google
      Cloud products as a transport intermediary. This field contains a
      reference to that transport intermediary. This information can be used
      for debugging purposes.
    uid: Output only. Server-assigned unique identifier for the trigger. The
      value is a UUID4 string and guaranteed to remain unchanged until the
      resource is deleted.
    updateTime: Output only. The last-modified time.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConditionsValue(_messages.Message):
        """Output only. The reason(s) why a trigger is in FAILED state.

    Messages:
      AdditionalProperty: An additional property for a ConditionsValue object.

    Fields:
      additionalProperties: Additional properties of type ConditionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConditionsValue object.

      Fields:
        key: Name of the additional property.
        value: A StateCondition attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('StateCondition', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User labels attached to the triggers that can be used to
    group resources.

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
    channel = _messages.StringField(1)
    conditions = _messages.MessageField('ConditionsValue', 2)
    createTime = _messages.StringField(3)
    destination = _messages.MessageField('Destination', 4)
    etag = _messages.StringField(5)
    eventDataContentType = _messages.StringField(6)
    eventFilters = _messages.MessageField('EventFilter', 7, repeated=True)
    labels = _messages.MessageField('LabelsValue', 8)
    name = _messages.StringField(9)
    serviceAccount = _messages.StringField(10)
    transport = _messages.MessageField('Transport', 11)
    uid = _messages.StringField(12)
    updateTime = _messages.StringField(13)