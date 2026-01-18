from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListApiServerSourcesResponse(_messages.Message):
    """ListApiServerSourcesResponse is a list of ApiServerSource resources.

  Fields:
    apiVersion: The API version for this call such as
      "sources.knative.dev/v1beta1".
    items: List of ApiServerSources.
    kind: The kind of this resource, in this case "ApiServerSourceList".
    metadata: Metadata associated with this ApiServerSource list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('ApiServerSource', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)