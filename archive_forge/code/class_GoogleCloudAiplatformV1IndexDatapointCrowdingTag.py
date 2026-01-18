from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1IndexDatapointCrowdingTag(_messages.Message):
    """Crowding tag is a constraint on a neighbor list produced by nearest
  neighbor search requiring that no more than some value k' of the k neighbors
  returned have the same value of crowding_attribute.

  Fields:
    crowdingAttribute: The attribute value used for crowding. The maximum
      number of neighbors to return per crowding attribute value
      (per_crowding_attribute_num_neighbors) is configured per-query. This
      field is ignored if per_crowding_attribute_num_neighbors is larger than
      the total number of neighbors to return for a given query.
  """
    crowdingAttribute = _messages.StringField(1)