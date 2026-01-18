from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SearchNearestEntitiesRequest(_messages.Message):
    """The request message for FeatureOnlineStoreService.SearchNearestEntities.

  Fields:
    query: Required. The query.
    returnFullEntity: Optional. If set to true, the full entities (including
      all vector values and metadata) of the nearest neighbors are returned;
      otherwise only entity id of the nearest neighbors will be returned. Note
      that returning full entities will significantly increase the latency and
      cost of the query.
  """
    query = _messages.MessageField('GoogleCloudAiplatformV1beta1NearestNeighborQuery', 1)
    returnFullEntity = _messages.BooleanField(2)