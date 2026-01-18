from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaListReplaysResponse(_messages.Message):
    """Response message for Simulator.ListReplays.

  Fields:
    nextPageToken: A token that you can use to retrieve the next page of
      results. If this field is omitted, there are no subsequent pages.
    replays: The list of Replay objects.
  """
    nextPageToken = _messages.StringField(1)
    replays = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaReplay', 2, repeated=True)