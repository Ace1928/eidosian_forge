from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ReadTensorboardUsageResponsePerUserUsageData(_messages.Message):
    """Per user usage data.

  Fields:
    username: User's username
    viewCount: Number of times the user has read data within the Tensorboard.
  """
    username = _messages.StringField(1)
    viewCount = _messages.IntegerField(2)