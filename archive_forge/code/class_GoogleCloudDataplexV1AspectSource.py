from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1AspectSource(_messages.Message):
    """AspectSource contains source system related information for the aspect.

  Fields:
    createTime: The create time of the aspect in the source system.
    updateTime: The update time of the aspect in the source system.
  """
    createTime = _messages.StringField(1)
    updateTime = _messages.StringField(2)