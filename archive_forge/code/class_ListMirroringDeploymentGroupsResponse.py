from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMirroringDeploymentGroupsResponse(_messages.Message):
    """Message for response to listing MirroringDeploymentGroups

  Fields:
    mirroringDeploymentGroups: The list of MirroringDeploymentGroup
    nextPageToken: A token identifying a page of results the server should
      return.
  """
    mirroringDeploymentGroups = _messages.MessageField('MirroringDeploymentGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)