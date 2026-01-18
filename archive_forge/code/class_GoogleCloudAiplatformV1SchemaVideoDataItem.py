from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaVideoDataItem(_messages.Message):
    """Payload of Video DataItem.

  Fields:
    gcsUri: Required. Google Cloud Storage URI points to the original video in
      user's bucket. The video is up to 50 GB in size and up to 3 hour in
      duration.
    mimeType: Output only. The mime type of the content of the video. Only the
      videos in below listed mime types are supported. Supported mime_type: -
      video/mp4 - video/avi - video/quicktime
  """
    gcsUri = _messages.StringField(1)
    mimeType = _messages.StringField(2)