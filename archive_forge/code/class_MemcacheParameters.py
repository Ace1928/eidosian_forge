from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemcacheParameters(_messages.Message):
    """A MemcacheParameters object.

  Messages:
    ParamsValue: User defined set of parameters to use in the memcached
      process.

  Fields:
    id: Output only. The unique ID associated with this set of parameters.
      Users can use this id to determine if the parameters associated with the
      instance differ from the parameters associated with the nodes. A
      discrepancy between parameter ids can inform users that they may need to
      take action to apply parameters on nodes.
    params: User defined set of parameters to use in the memcached process.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParamsValue(_messages.Message):
        """User defined set of parameters to use in the memcached process.

    Messages:
      AdditionalProperty: An additional property for a ParamsValue object.

    Fields:
      additionalProperties: Additional properties of type ParamsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParamsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    id = _messages.StringField(1)
    params = _messages.MessageField('ParamsValue', 2)