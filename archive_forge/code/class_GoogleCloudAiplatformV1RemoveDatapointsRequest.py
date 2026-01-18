from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1RemoveDatapointsRequest(_messages.Message):
    """Request message for IndexService.RemoveDatapoints

  Fields:
    datapointIds: A list of datapoint ids to be deleted.
  """
    datapointIds = _messages.StringField(1, repeated=True)