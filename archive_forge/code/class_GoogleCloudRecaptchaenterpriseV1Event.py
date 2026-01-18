from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1Event(_messages.Message):
    """The event being assessed.

  Enums:
    FraudPreventionValueValuesEnum: Optional. The Fraud Prevention setting for
      this assessment.

  Fields:
    expectedAction: Optional. The expected action for this type of event. This
      should be the same action provided at token generation time on client-
      side platforms already integrated with recaptcha enterprise.
    express: Optional. Flag for a reCAPTCHA express request for an assessment
      without a token. If enabled, `site_key` must reference a SCORE key with
      WAF feature set to EXPRESS.
    firewallPolicyEvaluation: Optional. Flag for enabling firewall policy
      config assessment. If this flag is enabled, the firewall policy will be
      evaluated and a suggested firewall action will be returned in the
      response.
    fraudPrevention: Optional. The Fraud Prevention setting for this
      assessment.
    hashedAccountId: Optional. Deprecated: use `user_info.account_id` instead.
      Unique stable hashed user identifier for the request. The identifier
      must be hashed using hmac-sha256 with stable secret.
    headers: Optional. HTTP header information about the request.
    ja3: Optional. JA3 fingerprint for SSL clients.
    requestedUri: Optional. The URI resource the user requested that triggered
      an assessment.
    siteKey: Optional. The site key that was used to invoke reCAPTCHA
      Enterprise on your site and generate the token.
    token: Optional. The user response token provided by the reCAPTCHA
      Enterprise client-side integration on your site.
    transactionData: Optional. Data describing a payment transaction to be
      assessed. Sending this data enables reCAPTCHA Enterprise Fraud
      Prevention and the FraudPreventionAssessment component in the response.
    userAgent: Optional. The user agent present in the request from the user's
      device related to this event.
    userInfo: Optional. Information about the user that generates this event,
      when they can be identified. They are often identified through the use
      of an account for logged-in requests or login/registration requests, or
      by providing user identifiers for guest actions like checkout.
    userIpAddress: Optional. The IP address in the request from the user's
      device related to this event.
    wafTokenAssessment: Optional. Flag for running WAF token assessment. If
      enabled, the token must be specified, and have been created by a WAF-
      enabled key.
  """

    class FraudPreventionValueValuesEnum(_messages.Enum):
        """Optional. The Fraud Prevention setting for this assessment.

    Values:
      FRAUD_PREVENTION_UNSPECIFIED: Default, unspecified setting. If opted in
        for automatic detection, `fraud_prevention_assessment` is returned
        based on the request. Otherwise, `fraud_prevention_assessment` is
        returned if `transaction_data` is present in the `Event` and Fraud
        Prevention is enabled in the Google Cloud console.
      ENABLED: Enable Fraud Prevention for this assessment, if Fraud
        Prevention is enabled in the Google Cloud console.
      DISABLED: Disable Fraud Prevention for this assessment, regardless of
        opt-in status or Google Cloud console settings.
    """
        FRAUD_PREVENTION_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
    expectedAction = _messages.StringField(1)
    express = _messages.BooleanField(2)
    firewallPolicyEvaluation = _messages.BooleanField(3)
    fraudPrevention = _messages.EnumField('FraudPreventionValueValuesEnum', 4)
    hashedAccountId = _messages.BytesField(5)
    headers = _messages.StringField(6, repeated=True)
    ja3 = _messages.StringField(7)
    requestedUri = _messages.StringField(8)
    siteKey = _messages.StringField(9)
    token = _messages.StringField(10)
    transactionData = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionData', 11)
    userAgent = _messages.StringField(12)
    userInfo = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1UserInfo', 13)
    userIpAddress = _messages.StringField(14)
    wafTokenAssessment = _messages.BooleanField(15)