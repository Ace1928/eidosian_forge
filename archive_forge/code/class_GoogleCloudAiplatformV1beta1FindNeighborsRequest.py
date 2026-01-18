from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FindNeighborsRequest(_messages.Message):
    """The request message for MatchService.FindNeighbors.

  Fields:
    deployedIndexId: The ID of the DeployedIndex that will serve the request.
      This request is sent to a specific IndexEndpoint, as per the
      IndexEndpoint.network. That IndexEndpoint also has
      IndexEndpoint.deployed_indexes, and each such index has a
      DeployedIndex.id field. The value of the field below must equal one of
      the DeployedIndex.id fields of the IndexEndpoint that is being called
      for this request.
    queries: The list of queries.
    returnFullDatapoint: If set to true, the full datapoints (including all
      vector values and restricts) of the nearest neighbors are returned. Note
      that returning full datapoint will significantly increase the latency
      and cost of the query.
  """
    deployedIndexId = _messages.StringField(1)
    queries = _messages.MessageField('GoogleCloudAiplatformV1beta1FindNeighborsRequestQuery', 2, repeated=True)
    returnFullDatapoint = _messages.BooleanField(3)