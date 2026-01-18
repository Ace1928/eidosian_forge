from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1Assessment(_messages.Message):
    """A reCAPTCHA Enterprise assessment resource.

  Fields:
    accountDefenderAssessment: Output only. Assessment returned by account
      defender when an account identifier is provided.
    accountVerification: Optional. Account verification information for
      identity verification. The assessment event must include a token and
      site key to use this feature.
    event: Optional. The event being assessed.
    firewallPolicyAssessment: Output only. Assessment returned when firewall
      policies belonging to the project are evaluated using the field
      firewall_policy_evaluation.
    fraudPreventionAssessment: Output only. Assessment returned by Fraud
      Prevention when TransactionData is provided.
    fraudSignals: Output only. Fraud Signals specific to the users involved in
      a payment transaction.
    name: Output only. Identifier. The resource name for the Assessment in the
      format `projects/{project}/assessments/{assessment}`.
    privatePasswordLeakVerification: Optional. The private password leak
      verification field contains the parameters that are used to to check for
      leaks privately without sharing user credentials.
    riskAnalysis: Output only. The risk analysis result for the event being
      assessed.
    tokenProperties: Output only. Properties of the provided event token.
  """
    accountDefenderAssessment = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1AccountDefenderAssessment', 1)
    accountVerification = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1AccountVerificationInfo', 2)
    event = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1Event', 3)
    firewallPolicyAssessment = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallPolicyAssessment', 4)
    fraudPreventionAssessment = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessment', 5)
    fraudSignals = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudSignals', 6)
    name = _messages.StringField(7)
    privatePasswordLeakVerification = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1PrivatePasswordLeakVerification', 8)
    riskAnalysis = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1RiskAnalysis', 9)
    tokenProperties = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TokenProperties', 10)