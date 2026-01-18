from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAttributeDefinitionsResponse(_messages.Message):
    """A ListAttributeDefinitionsResponse object.

  Fields:
    attributeDefinitions: The returned Attribute definitions. The maximum
      number of attributes returned is determined by the value of page_size in
      the ListAttributeDefinitionsRequest.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    attributeDefinitions = _messages.MessageField('AttributeDefinition', 1, repeated=True)
    nextPageToken = _messages.StringField(2)