from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PathValueMatchersValue(_messages.Message):
    """Similar to path_filters, this contains set of filters to apply if
    `path` field refers to array elements. This is meant to support value
    matching beyond exact match. To perform exact match, use path_filters.
    When both path_filters and path_value_matchers are set, an implicit AND
    must be performed.

    Messages:
      AdditionalProperty: An additional property for a PathValueMatchersValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PathValueMatchersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PathValueMatchersValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudRecommenderV1alpha2ValueMatcher attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudRecommenderV1alpha2ValueMatcher', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)