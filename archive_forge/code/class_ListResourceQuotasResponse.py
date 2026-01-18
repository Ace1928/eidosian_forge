from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListResourceQuotasResponse(_messages.Message):
    """List of fleet namespaces.

  Fields:
    nextPageToken: A token to request the next page of resources from the
      `ListNamespaces` method. The value of an empty string means that there
      are no more resources to return.
    resourceQuotas: The list of fleet namespaces
  """
    nextPageToken = _messages.StringField(1)
    resourceQuotas = _messages.MessageField('ResourceQuota', 2, repeated=True)