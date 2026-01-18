from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUsableWorkstationsResponse(_messages.Message):
    """Response message for ListUsableWorkstations.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachable: Unreachable resources.
    workstations: The requested workstations.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    workstations = _messages.MessageField('Workstation', 3, repeated=True)