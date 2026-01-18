from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LaunchOptionsValue(_messages.Message):
    """Launch options for this Flex Template job. This is a common set of
    options across languages and templates. This should not be used to pass
    job parameters.

    Messages:
      AdditionalProperty: An additional property for a LaunchOptionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type LaunchOptionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LaunchOptionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)