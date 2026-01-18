from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NearestNeighbors(_messages.Message):
    """Nearest neighbors for one query.

  Fields:
    neighbors: All its neighbors.
  """
    neighbors = _messages.MessageField('GoogleCloudAiplatformV1NearestNeighborsNeighbor', 1, repeated=True)