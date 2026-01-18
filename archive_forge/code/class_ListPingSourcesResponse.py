from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListPingSourcesResponse(_messages.Message):
    """ListPingSourcesResponse is a list of PingSource resources.

  Fields:
    apiVersion: The API version for this call such as
      "sources.knative.dev/v1beta1".
    items: List of PingSources.
    kind: The kind of this resource, in this case "PingSourceList".
    metadata: Metadata associated with this PingSource list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('PingSource', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)