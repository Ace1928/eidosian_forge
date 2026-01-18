from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LimitsValue(_messages.Message):
    """Only `memory` and `cpu` keys in the map are supported. Notes: * The
    only supported values for CPU are '1', '2', '4', and '8'. Setting 4 CPU
    requires at least 2Gi of memory. For more information, go to
    https://cloud.google.com/run/docs/configuring/cpu. * For supported
    'memory' values and syntax, go to
    https://cloud.google.com/run/docs/configuring/memory-limits

    Messages:
      AdditionalProperty: An additional property for a LimitsValue object.

    Fields:
      additionalProperties: Additional properties of type LimitsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LimitsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)