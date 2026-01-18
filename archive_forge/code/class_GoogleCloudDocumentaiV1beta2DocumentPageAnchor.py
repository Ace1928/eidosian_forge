from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageAnchor(_messages.Message):
    """Referencing the visual context of the entity in the Document.pages. Page
  anchors can be cross-page, consist of multiple bounding polygons and
  optionally reference specific layout element types.

  Fields:
    pageRefs: One or more references to visual page elements
  """
    pageRefs = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentPageAnchorPageRef', 1, repeated=True)