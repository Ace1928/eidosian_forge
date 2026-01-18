from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FirewallActionRedirectAction(_messages.Message):
    """A redirect action returns a 307 (temporary redirect) response, pointing
  the user to a ReCaptcha interstitial page to attach a token.
  """