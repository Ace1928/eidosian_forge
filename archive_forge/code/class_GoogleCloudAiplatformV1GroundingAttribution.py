from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GroundingAttribution(_messages.Message):
    """Grounding attribution.

  Fields:
    confidenceScore: Optional. Output only. Confidence score of the
      attribution. Ranges from 0 to 1. 1 is the most confident.
    segment: Output only. Segment of the content this attribution belongs to.
    web: Optional. Attribution from the web.
  """
    confidenceScore = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    segment = _messages.MessageField('GoogleCloudAiplatformV1Segment', 2)
    web = _messages.MessageField('GoogleCloudAiplatformV1GroundingAttributionWeb', 3)