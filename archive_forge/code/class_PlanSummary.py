from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PlanSummary(_messages.Message):
    """Planning phase information for the query.

  Messages:
    IndexesUsedValueListEntry: A IndexesUsedValueListEntry object.

  Fields:
    indexesUsed: The indexes selected for the query. For example: [
      {"query_scope": "Collection", "properties": "(foo ASC, __name__ ASC)"},
      {"query_scope": "Collection", "properties": "(bar ASC, __name__ ASC)"} ]
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class IndexesUsedValueListEntry(_messages.Message):
        """A IndexesUsedValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a
        IndexesUsedValueListEntry object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a IndexesUsedValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    indexesUsed = _messages.MessageField('IndexesUsedValueListEntry', 1, repeated=True)