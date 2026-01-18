from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NearestNeighborQuery(_messages.Message):
    """A query to find a number of similar entities.

  Fields:
    embedding: Optional. The embedding vector that be used for similar search.
    entityId: Optional. The entity id whose similar entities should be
      searched for. If embedding is set, search will use embedding instead of
      entity_id.
    neighborCount: Optional. The number of similar entities to be retrieved
      from feature view for each query.
    parameters: Optional. Parameters that can be set to tune query on the fly.
    perCrowdingAttributeNeighborCount: Optional. Crowding is a constraint on a
      neighbor list produced by nearest neighbor search requiring that no more
      than sper_crowding_attribute_neighbor_count of the k neighbors returned
      have the same value of crowding_attribute. It's used for improving
      result diversity.
    stringFilters: Optional. The list of string filters.
  """
    embedding = _messages.MessageField('GoogleCloudAiplatformV1beta1NearestNeighborQueryEmbedding', 1)
    entityId = _messages.StringField(2)
    neighborCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1beta1NearestNeighborQueryParameters', 4)
    perCrowdingAttributeNeighborCount = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    stringFilters = _messages.MessageField('GoogleCloudAiplatformV1beta1NearestNeighborQueryStringFilter', 6, repeated=True)