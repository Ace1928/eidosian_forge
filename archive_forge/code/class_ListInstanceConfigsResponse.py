from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstanceConfigsResponse(_messages.Message):
    """The response for ListInstanceConfigs.

  Fields:
    instanceConfigs: The list of requested instance configurations.
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListInstanceConfigs call to fetch more of the matching instance
      configurations.
  """
    instanceConfigs = _messages.MessageField('InstanceConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)