from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMachinesResponse(_messages.Message):
    """List of machines in a site.

  Fields:
    machines: Machines in the site.
    nextPageToken: A token to retrieve next page of results.
    unreachable: Locations that could not be reached.
  """
    machines = _messages.MessageField('Machine', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)