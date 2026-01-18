from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AnnotationLabelsValue(_messages.Message):
    """Labels that will be applied to newly imported Annotations. If two
    Annotations are identical, one of them will be deduped. Two Annotations
    are considered identical if their payload, payload_schema_uri and all of
    their labels are the same. These labels will be overridden by Annotation
    labels specified inside index file referenced by import_schema_uri, e.g.
    jsonl file.

    Messages:
      AdditionalProperty: An additional property for a AnnotationLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AnnotationLabelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AnnotationLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)