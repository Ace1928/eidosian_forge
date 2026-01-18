from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentEntity(_messages.Message):
    """An entity that could be a phrase in the text or a property that belongs
  to the document. It is a known entity type, such as a person, an
  organization, or location.

  Fields:
    confidence: Optional. Confidence of detected Schema entity. Range `[0,
      1]`.
    id: Optional. Canonical id. This will be a unique value in the entity list
      for this document.
    mentionId: Optional. Deprecated. Use `id` field instead.
    mentionText: Optional. Text value of the entity e.g. `1600 Amphitheatre
      Pkwy`.
    normalizedValue: Optional. Normalized entity value. Absent if the
      extracted value could not be converted or the type (e.g. address) is not
      supported for certain parsers. This field is also only populated for
      certain supported document types.
    pageAnchor: Optional. Represents the provenance of this entity wrt. the
      location on the page where it was found.
    properties: Optional. Entities can be nested to form a hierarchical data
      structure representing the content in the document.
    provenance: Optional. The history of this annotation.
    redacted: Optional. Whether the entity will be redacted for de-
      identification purposes.
    textAnchor: Optional. Provenance of the entity. Text anchor indexing into
      the Document.text.
    type: Required. Entity type from a schema e.g. `Address`.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    id = _messages.StringField(2)
    mentionId = _messages.StringField(3)
    mentionText = _messages.StringField(4)
    normalizedValue = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentEntityNormalizedValue', 5)
    pageAnchor = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageAnchor', 6)
    properties = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentEntity', 7, repeated=True)
    provenance = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentProvenance', 8)
    redacted = _messages.BooleanField(9)
    textAnchor = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentTextAnchor', 10)
    type = _messages.StringField(11)