from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Score(_messages.Message):
    """Represents Security Score.

  Fields:
    component: Component containing score, recommendations and actions.
    subcomponents: List of all the drilldown score components.
    timeRange: Start and end time for the score.
  """
    component = _messages.MessageField('GoogleCloudApigeeV1ScoreComponent', 1)
    subcomponents = _messages.MessageField('GoogleCloudApigeeV1ScoreComponent', 2, repeated=True)
    timeRange = _messages.MessageField('GoogleTypeInterval', 3)