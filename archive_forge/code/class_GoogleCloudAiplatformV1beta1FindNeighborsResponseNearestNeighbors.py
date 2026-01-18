from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FindNeighborsResponseNearestNeighbors(_messages.Message):
    """Nearest neighbors for one query.

  Fields:
    id: The ID of the query datapoint.
    neighbors: All its neighbors.
  """
    id = _messages.StringField(1)
    neighbors = _messages.MessageField('GoogleCloudAiplatformV1beta1FindNeighborsResponseNeighbor', 2, repeated=True)