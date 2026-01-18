from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NearestNeighborsNeighbor(_messages.Message):
    """A neighbor of the query vector.

  Fields:
    distance: The distance between the neighbor and the query vector.
    entityId: The id of the similar entity.
    entityKeyValues: The attributes of the neighbor, e.g. filters, crowding
      and metadata Note that full entities are returned only when
      "return_full_entity" is set to true. Otherwise, only the "entity_id" and
      "distance" fields are populated.
  """
    distance = _messages.FloatField(1)
    entityId = _messages.StringField(2)
    entityKeyValues = _messages.MessageField('GoogleCloudAiplatformV1FetchFeatureValuesResponse', 3)