from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FindUnregisteredWorkloadsResponse(_messages.Message):
    """Response for FindUnregisteredWorkloads.

  Fields:
    discoveredWorkloads: List of Discovered Workloads.
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    discoveredWorkloads = _messages.MessageField('DiscoveredWorkload', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)