from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReadIndexDatapointsRequest(_messages.Message):
    """The request message for MatchService.ReadIndexDatapoints.

  Fields:
    deployedIndexId: The ID of the DeployedIndex that will serve the request.
    ids: IDs of the datapoints to be searched for.
  """
    deployedIndexId = _messages.StringField(1)
    ids = _messages.StringField(2, repeated=True)