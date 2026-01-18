from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1Key(_messages.Message):
    """A key used to identify and configure applications (web and/or mobile)
  that use reCAPTCHA Enterprise.

  Messages:
    LabelsValue: Optional. See [Creating and managing labels]
      (https://cloud.google.com/recaptcha-enterprise/docs/labels).

  Fields:
    androidSettings: Settings for keys that can be used by Android apps.
    createTime: Output only. The timestamp corresponding to the creation of
      this key.
    displayName: Required. Human-readable display name of this key. Modifiable
      by user.
    iosSettings: Settings for keys that can be used by iOS apps.
    labels: Optional. See [Creating and managing labels]
      (https://cloud.google.com/recaptcha-enterprise/docs/labels).
    name: Identifier. The resource name for the Key in the format
      `projects/{project}/keys/{key}`.
    testingOptions: Optional. Options for user acceptance testing.
    wafSettings: Optional. Settings for WAF
    webSettings: Settings for keys that can be used by websites.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. See [Creating and managing labels]
    (https://cloud.google.com/recaptcha-enterprise/docs/labels).

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
    androidSettings = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1AndroidKeySettings', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    iosSettings = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1IOSKeySettings', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    testingOptions = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TestingOptions', 7)
    wafSettings = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1WafSettings', 8)
    webSettings = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1WebKeySettings', 9)