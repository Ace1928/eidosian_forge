from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPowerSSHKeysResponse(_messages.Message):
    """Response message containing the list of Power SSH Keys.

  Fields:
    nextPageToken: A token identifying a page of results from the server.
    powerSshKeys: The list of SSH Keys.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    powerSshKeys = _messages.MessageField('PowerSSHKey', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)