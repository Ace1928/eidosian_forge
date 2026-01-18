from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListResourcesResponse(_messages.Message):
    """A response to a 'ListResources' call. Contains a list of Resources.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      'ListResources' method. The value of an empty string means that there
      are no more resources to return.
    resources: List of Resourcess.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    resources = _messages.MessageField('Resource', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)