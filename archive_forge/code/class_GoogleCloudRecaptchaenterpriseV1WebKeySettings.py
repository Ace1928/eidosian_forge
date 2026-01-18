from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1WebKeySettings(_messages.Message):
    """Settings specific to keys that can be used by websites.

  Enums:
    ChallengeSecurityPreferenceValueValuesEnum: Optional. Settings for the
      frequency and difficulty at which this key triggers captcha challenges.
      This should only be specified for IntegrationTypes CHECKBOX and
      INVISIBLE.
    IntegrationTypeValueValuesEnum: Required. Describes how this key is
      integrated with the website.

  Fields:
    allowAllDomains: Optional. If set to true, it means allowed_domains will
      not be enforced.
    allowAmpTraffic: Optional. If set to true, the key can be used on AMP
      (Accelerated Mobile Pages) websites. This is supported only for the
      SCORE integration type.
    allowedDomains: Optional. Domains or subdomains of websites allowed to use
      the key. All subdomains of an allowed domain are automatically allowed.
      A valid domain requires a host and must not include any path, port,
      query or fragment. Examples: 'example.com' or 'subdomain.example.com'
    challengeSecurityPreference: Optional. Settings for the frequency and
      difficulty at which this key triggers captcha challenges. This should
      only be specified for IntegrationTypes CHECKBOX and INVISIBLE.
    integrationType: Required. Describes how this key is integrated with the
      website.
  """

    class ChallengeSecurityPreferenceValueValuesEnum(_messages.Enum):
        """Optional. Settings for the frequency and difficulty at which this key
    triggers captcha challenges. This should only be specified for
    IntegrationTypes CHECKBOX and INVISIBLE.

    Values:
      CHALLENGE_SECURITY_PREFERENCE_UNSPECIFIED: Default type that indicates
        this enum hasn't been specified.
      USABILITY: Key tends to show fewer and easier challenges.
      BALANCE: Key tends to show balanced (in amount and difficulty)
        challenges.
      SECURITY: Key tends to show more and harder challenges.
    """
        CHALLENGE_SECURITY_PREFERENCE_UNSPECIFIED = 0
        USABILITY = 1
        BALANCE = 2
        SECURITY = 3

    class IntegrationTypeValueValuesEnum(_messages.Enum):
        """Required. Describes how this key is integrated with the website.

    Values:
      INTEGRATION_TYPE_UNSPECIFIED: Default type that indicates this enum
        hasn't been specified. This is not a valid IntegrationType, one of the
        other types must be specified instead.
      SCORE: Only used to produce scores. It doesn't display the "I'm not a
        robot" checkbox and never shows captcha challenges.
      CHECKBOX: Displays the "I'm not a robot" checkbox and may show captcha
        challenges after it is checked.
      INVISIBLE: Doesn't display the "I'm not a robot" checkbox, but may show
        captcha challenges after risk analysis.
    """
        INTEGRATION_TYPE_UNSPECIFIED = 0
        SCORE = 1
        CHECKBOX = 2
        INVISIBLE = 3
    allowAllDomains = _messages.BooleanField(1)
    allowAmpTraffic = _messages.BooleanField(2)
    allowedDomains = _messages.StringField(3, repeated=True)
    challengeSecurityPreference = _messages.EnumField('ChallengeSecurityPreferenceValueValuesEnum', 4)
    integrationType = _messages.EnumField('IntegrationTypeValueValuesEnum', 5)