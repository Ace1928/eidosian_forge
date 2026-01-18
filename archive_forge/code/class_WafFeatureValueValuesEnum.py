from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WafFeatureValueValuesEnum(_messages.Enum):
    """Required. The WAF feature for which this key is enabled.

    Values:
      WAF_FEATURE_UNSPECIFIED: Undefined feature.
      CHALLENGE_PAGE: Redirects suspicious traffic to reCAPTCHA.
      SESSION_TOKEN: Use reCAPTCHA session-tokens to protect the whole user
        session on the site's domain.
      ACTION_TOKEN: Use reCAPTCHA action-tokens to protect user actions.
      EXPRESS: Use reCAPTCHA WAF express protection to protect any content
        other than web pages, like APIs and IoT devices.
    """
    WAF_FEATURE_UNSPECIFIED = 0
    CHALLENGE_PAGE = 1
    SESSION_TOKEN = 2
    ACTION_TOKEN = 3
    EXPRESS = 4