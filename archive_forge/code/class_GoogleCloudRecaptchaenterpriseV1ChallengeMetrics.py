from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1ChallengeMetrics(_messages.Message):
    """Metrics related to challenges.

  Fields:
    failedCount: Count of submitted challenge solutions that were incorrect or
      otherwise deemed suspicious such that a subsequent challenge was
      triggered.
    nocaptchaCount: Count of nocaptchas (successful verification without a
      challenge) issued.
    pageloadCount: Count of reCAPTCHA checkboxes or badges rendered. This is
      mostly equivalent to a count of pageloads for pages that include
      reCAPTCHA.
    passedCount: Count of nocaptchas (successful verification without a
      challenge) plus submitted challenge solutions that were correct and
      resulted in verification.
  """
    failedCount = _messages.IntegerField(1)
    nocaptchaCount = _messages.IntegerField(2)
    pageloadCount = _messages.IntegerField(3)
    passedCount = _messages.IntegerField(4)