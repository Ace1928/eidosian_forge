from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ParamTypesValue(_messages.Message):
    """It is not always possible for Cloud Spanner to infer the right SQL
    type from a JSON value. For example, values of type `BYTES` and values of
    type `STRING` both appear in params as JSON strings. In these cases,
    `param_types` can be used to specify the exact SQL type for some or all of
    the SQL statement parameters. See the definition of Type for more
    information about SQL types.

    Messages:
      AdditionalProperty: An additional property for a ParamTypesValue object.

    Fields:
      additionalProperties: Additional properties of type ParamTypesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ParamTypesValue object.

      Fields:
        key: Name of the additional property.
        value: A Type attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Type', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)