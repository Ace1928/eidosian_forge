from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunConfig(_messages.Message):
    """CloudRunConfig contains the Cloud Run runtime configuration.

  Fields:
    automaticTrafficControl: Whether Cloud Deploy should update the traffic
      stanza in a Cloud Run Service on the user's behalf to facilitate traffic
      splitting. This is required to be true for CanaryDeployments, but
      optional for CustomCanaryDeployments.
    canaryRevisionTags: Optional. A list of tags that are added to the canary
      revision while the canary phase is in progress.
    priorRevisionTags: Optional. A list of tags that are added to the prior
      revision while the canary phase is in progress.
    stableRevisionTags: Optional. A list of tags that are added to the final
      stable revision when the stable phase is applied.
  """
    automaticTrafficControl = _messages.BooleanField(1)
    canaryRevisionTags = _messages.StringField(2, repeated=True)
    priorRevisionTags = _messages.StringField(3, repeated=True)
    stableRevisionTags = _messages.StringField(4, repeated=True)