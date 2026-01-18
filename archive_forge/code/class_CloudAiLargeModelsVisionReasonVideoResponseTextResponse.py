from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionReasonVideoResponseTextResponse(_messages.Message):
    """Contains text that is the response of the video captioning.

  Fields:
    relativeTemporalPartition: Partition of the caption's video in time. This
      field is intended for video captioning. To represent the start time and
      end time of the caption's video.
    text: Text information
  """
    relativeTemporalPartition = _messages.MessageField('CloudAiLargeModelsVisionRelativeTemporalPartition', 1)
    text = _messages.StringField(2)