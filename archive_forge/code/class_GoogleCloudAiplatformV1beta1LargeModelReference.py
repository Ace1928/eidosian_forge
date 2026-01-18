from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1LargeModelReference(_messages.Message):
    """Contains information about the Large Model.

  Fields:
    name: Required. The unique name of the large Foundation or pre-built
      model. Like "chat-bison", "text-bison". Or model name with version ID,
      like "chat-bison@001", "text-bison@005", etc.
  """
    name = _messages.StringField(1)