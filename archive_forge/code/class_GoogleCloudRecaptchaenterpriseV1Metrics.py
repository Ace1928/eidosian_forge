from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1Metrics(_messages.Message):
    """Metrics for a single Key.

  Fields:
    challengeMetrics: Metrics will be continuous and in order by dates, and in
      the granularity of day. Only challenge-based keys (CHECKBOX, INVISIBLE),
      will have challenge-based data.
    name: Output only. Identifier. The name of the metrics, in the format
      `projects/{project}/keys/{key}/metrics`.
    scoreMetrics: Metrics will be continuous and in order by dates, and in the
      granularity of day. All Key types should have score-based data.
    startTime: Inclusive start time aligned to a day (UTC).
  """
    challengeMetrics = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1ChallengeMetrics', 1, repeated=True)
    name = _messages.StringField(2)
    scoreMetrics = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1ScoreMetrics', 3, repeated=True)
    startTime = _messages.StringField(4)