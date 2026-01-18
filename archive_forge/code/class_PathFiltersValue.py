from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PathFiltersValue(_messages.Message):
    """Set of filters to apply if `path` refers to array elements or nested
    array elements in order to narrow down to a single unique element that is
    being tested/modified. This is intended to be an exact match per filter.
    To perform advanced matching, use path_value_matchers. * Example: ``` {
    "/versions/*/name" : "it-123" "/versions/*/targetSize/percent": 20 } ``` *
    Example: ``` { "/bindings/*/role": "roles/owner" "/bindings/*/condition" :
    null } ``` * Example: ``` { "/bindings/*/role": "roles/owner"
    "/bindings/*/members/*" : ["x@example.com", "y@example.com"] } ``` When
    both path_filters and path_value_matchers are set, an implicit AND must be
    performed.

    Messages:
      AdditionalProperty: An additional property for a PathFiltersValue
        object.

    Fields:
      additionalProperties: Additional properties of type PathFiltersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PathFiltersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)