from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNamespacesResponse(_messages.Message):
    """List of fleet namespaces.

  Fields:
    namespaces: The list of fleet namespaces
    nextPageToken: A token to request the next page of resources from the
      `ListNamespaces` method. The value of an empty string means that there
      are no more resources to return.
  """
    namespaces = _messages.MessageField('Namespace', 1, repeated=True)
    nextPageToken = _messages.StringField(2)