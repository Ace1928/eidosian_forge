from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentLabel(_messages.Message):
    """Label attaches schema information and/or other metadata to segments
  within a Document. Multiple Labels on a single field can denote either
  different labels, different instances of the same label created at different
  times, or some combination of both.

  Fields:
    automlModel: Label is generated AutoML model. This field stores the full
      resource name of the AutoML model. Format: `projects/{project-
      id}/locations/{location-id}/models/{model-id}`
    confidence: Confidence score between 0 and 1 for label assignment.
    name: Name of the label. When the label is generated from AutoML Text
      Classification model, this field represents the name of the category.
  """
    automlModel = _messages.StringField(1)
    confidence = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    name = _messages.StringField(3)