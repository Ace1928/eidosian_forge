from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class AttributesValue(_messages.Message):
    """Endpoint configuration attributes.  Every endpoint has a set of API
    supported attributes that can be used to control different aspects of the
    message delivery.  The currently supported attribute is `x-goog-version`,
    which you can use to change the format of the pushed message. This
    attribute indicates the version of the data expected by the endpoint. This
    controls the shape of the pushed message (i.e., its fields and metadata).
    The endpoint version is based on the version of the Pub/Sub API.  If not
    present during the `CreateSubscription` call, it will default to the
    version of the API used to make such call. If not present during a
    `ModifyPushConfig` call, its value will not be changed. `GetSubscription`
    calls will always return a valid version, even if the subscription was
    created without this attribute.  The possible values for this attribute
    are:  * `v1beta1`: uses the push format defined in the v1beta1 Pub/Sub
    API. * `v1` or `v1beta2`: uses the push format defined in the v1 Pub/Sub
    API.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)