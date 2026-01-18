from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListCustomResourceDefinitionsResponse(_messages.Message):
    """A ListCustomResourceDefinitionsResponse object.

  Fields:
    apiVersion: The API version for this call such as
      "k8s.apiextensions.io/v1".
    items: List of CustomResourceDefinitions.
    kind: The kind of this resource, in this case
      "CustomResourceDefinitionList".
    metadata: Metadata associated with this CustomResourceDefinition list.
    unreachable: Locations that could not be reached.
  """
    apiVersion = _messages.StringField(1)
    items = _messages.MessageField('CustomResourceDefinition', 2, repeated=True)
    kind = _messages.StringField(3)
    metadata = _messages.MessageField('ListMeta', 4)
    unreachable = _messages.StringField(5, repeated=True)