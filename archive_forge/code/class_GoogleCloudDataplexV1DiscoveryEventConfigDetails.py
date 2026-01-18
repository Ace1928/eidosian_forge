from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DiscoveryEventConfigDetails(_messages.Message):
    """Details about configuration events.

  Messages:
    ParametersValue: A list of discovery configuration parameters in effect.
      The keys are the field paths within DiscoverySpec. Eg. includePatterns,
      excludePatterns, csvOptions.disableTypeInference, etc.

  Fields:
    parameters: A list of discovery configuration parameters in effect. The
      keys are the field paths within DiscoverySpec. Eg. includePatterns,
      excludePatterns, csvOptions.disableTypeInference, etc.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """A list of discovery configuration parameters in effect. The keys are
    the field paths within DiscoverySpec. Eg. includePatterns,
    excludePatterns, csvOptions.disableTypeInference, etc.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    parameters = _messages.MessageField('ParametersValue', 1)