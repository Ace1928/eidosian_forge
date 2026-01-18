from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingStageLocation(_messages.Message):
    """Identifies the location of a streaming computation stage, for stage-to-
  stage communication.

  Fields:
    streamId: Identifies the particular stream within the streaming Dataflow
      job.
  """
    streamId = _messages.StringField(1)