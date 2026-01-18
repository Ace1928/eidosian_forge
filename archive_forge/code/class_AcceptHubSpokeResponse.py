from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AcceptHubSpokeResponse(_messages.Message):
    """The response for HubService.AcceptHubSpoke.

  Fields:
    spoke: The spoke that was operated on.
  """
    spoke = _messages.MessageField('Spoke', 1)