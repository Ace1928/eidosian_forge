from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ScriptVariablesValue(_messages.Message):
    """Optional. Mapping of query variable names to values (equivalent to the
    Spark SQL command: SET `name="value";`).

    Messages:
      AdditionalProperty: An additional property for a ScriptVariablesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ScriptVariablesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ScriptVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)