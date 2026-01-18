from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FindNeighborsResponseNeighbor(_messages.Message):
    """A neighbor of the query vector.

  Fields:
    datapoint: The datapoint of the neighbor. Note that full datapoints are
      returned only when "return_full_datapoint" is set to true. Otherwise,
      only the "datapoint_id" and "crowding_tag" fields are populated.
    distance: The distance between the neighbor and the query vector.
  """
    datapoint = _messages.MessageField('GoogleCloudAiplatformV1beta1IndexDatapoint', 1)
    distance = _messages.FloatField(2)