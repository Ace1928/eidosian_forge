from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlatformPolicy(_messages.Message):
    """A Binary Authorization platform policy for deployments on various
  platforms.

  Fields:
    cloudRunPolicy: Optional. Cloud Run platform-specific policy.
    description: Optional. A description comment about the policy.
    gkePolicy: Optional. GKE platform-specific policy.
    name: Output only. The relative resource name of the Binary Authorization
      platform policy, in the form of `projects/*/platforms/*/policies/*`.
    updateTime: Output only. Time when the policy was last updated.
  """
    cloudRunPolicy = _messages.MessageField('InlineCloudRunPolicy', 1)
    description = _messages.StringField(2)
    gkePolicy = _messages.MessageField('GkePolicy', 3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)