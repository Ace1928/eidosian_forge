from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkSet(_messages.Message):
    """A LinkSet object.

  Messages:
    LinksValue: Mapping of a qualifier to an asset link. Asset Link Format:
      "p/p/l/l/assets/"

  Fields:
    links: Mapping of a qualifier to an asset link. Asset Link Format:
      "p/p/l/l/assets/"
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LinksValue(_messages.Message):
        """Mapping of a qualifier to an asset link. Asset Link Format:
    "p/p/l/l/assets/"

    Messages:
      AdditionalProperty: An additional property for a LinksValue object.

    Fields:
      additionalProperties: Additional properties of type LinksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LinksValue object.

      Fields:
        key: Name of the additional property.
        value: A Asset attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Asset', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    links = _messages.MessageField('LinksValue', 1)