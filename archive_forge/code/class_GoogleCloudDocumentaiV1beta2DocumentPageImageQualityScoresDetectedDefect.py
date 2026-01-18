from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentPageImageQualityScoresDetectedDefect(_messages.Message):
    """Image Quality Defects

  Fields:
    confidence: Confidence of detected defect. Range `[0, 1]` where `1`
      indicates strong confidence that the defect exists.
    type: Name of the defect type. Supported values are: -
      `quality/defect_blurry` - `quality/defect_noisy` - `quality/defect_dark`
      - `quality/defect_faint` - `quality/defect_text_too_small` -
      `quality/defect_document_cutoff` - `quality/defect_text_cutoff` -
      `quality/defect_glare`
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    type = _messages.StringField(2)