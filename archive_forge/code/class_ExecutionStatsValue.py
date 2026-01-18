from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ExecutionStatsValue(_messages.Message):
    """The execution statistics associated with the node, contained in a
    group of key-value pairs. Only present if the plan was returned as a
    result of a profile query. For example, number of executions, number of
    rows/time per execution etc.

    Messages:
      AdditionalProperty: An additional property for a ExecutionStatsValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ExecutionStatsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)