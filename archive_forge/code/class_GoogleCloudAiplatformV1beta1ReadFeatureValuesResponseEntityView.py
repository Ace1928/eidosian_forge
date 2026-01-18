from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReadFeatureValuesResponseEntityView(_messages.Message):
    """Entity view with Feature values.

  Fields:
    data: Each piece of data holds the k requested values for one requested
      Feature. If no values for the requested Feature exist, the corresponding
      cell will be empty. This has the same size and is in the same order as
      the features from the header ReadFeatureValuesResponse.header.
    entityId: ID of the requested entity.
  """
    data = _messages.MessageField('GoogleCloudAiplatformV1beta1ReadFeatureValuesResponseEntityViewData', 1, repeated=True)
    entityId = _messages.StringField(2)