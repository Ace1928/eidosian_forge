from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2RevisionScalingStatus(_messages.Message):
    """Effective settings for the current revision

  Fields:
    desiredMinInstanceCount: The current number of min instances provisioned
      for this revision.
  """
    desiredMinInstanceCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)