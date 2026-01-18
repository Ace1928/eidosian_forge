from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ModulesValue(_messages.Message):
    """The configurations including the state of enablement for the service's
    different modules. The absence of a module in the map implies its
    configuration is inherited from its parent's.

    Messages:
      AdditionalProperty: An additional property for a ModulesValue object.

    Fields:
      additionalProperties: Additional properties of type ModulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ModulesValue object.

      Fields:
        key: Name of the additional property.
        value: A Config attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Config', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)