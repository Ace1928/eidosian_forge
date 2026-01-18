from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaVisualInspectionClassificationLabelSavedQueryMetadata(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaVisualInspectionClassificationLabelSavedQ
  ueryMetadata object.

  Fields:
    multiLabel: Whether or not the classification label is multi_label.
  """
    multiLabel = _messages.BooleanField(1)