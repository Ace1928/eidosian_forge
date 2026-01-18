from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AdditionalUserLabelsValue(_messages.Message):
    """Additional user labels to be specified for the job. Keys and values
    should follow the restrictions specified in the [labeling
    restrictions](https://cloud.google.com/compute/docs/labeling-
    resources#restrictions) page. An object containing a list of key/value
    pairs. Example: { "name": "wrench", "mass": "1kg", "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a
        AdditionalUserLabelsValue object.

    Fields:
      additionalProperties: Additional properties of type
        AdditionalUserLabelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AdditionalUserLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)