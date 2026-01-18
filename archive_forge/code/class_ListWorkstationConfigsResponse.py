from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkstationConfigsResponse(_messages.Message):
    """Response message for ListWorkstationConfigs.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachable: Unreachable resources.
    workstationConfigs: The requested configs.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    workstationConfigs = _messages.MessageField('WorkstationConfig', 3, repeated=True)