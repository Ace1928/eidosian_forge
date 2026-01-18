from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAttackPathsResponse(_messages.Message):
    """Response message for listing the attack paths for a given simulation or
  valued resource.

  Fields:
    attackPaths: The attack paths that the attack path simulation identified.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results.
  """
    attackPaths = _messages.MessageField('AttackPath', 1, repeated=True)
    nextPageToken = _messages.StringField(2)