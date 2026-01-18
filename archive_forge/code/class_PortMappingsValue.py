from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PortMappingsValue(_messages.Message):
    """The container-to-host port mappings installed for this container. This
    set will contain any ports exposed using the `PUBLISH_EXPOSED_PORTS` flag
    as well as any specified in the `Action` definition.

    Messages:
      AdditionalProperty: An additional property for a PortMappingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type PortMappingsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PortMappingsValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)