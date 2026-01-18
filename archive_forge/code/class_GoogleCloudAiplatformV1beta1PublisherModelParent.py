from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelParent(_messages.Message):
    """The information about the parent of a model.

  Fields:
    displayName: Required. The display name of the parent. E.g., LaMDA, T5,
      Vision API, Natural Language API.
    reference: Optional. The Google Cloud resource name or the URI reference.
  """
    displayName = _messages.StringField(1)
    reference = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelResourceReference', 2)