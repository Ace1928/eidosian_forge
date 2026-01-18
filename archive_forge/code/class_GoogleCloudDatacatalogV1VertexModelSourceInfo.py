from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1VertexModelSourceInfo(_messages.Message):
    """Detail description of the source information of a Vertex model.

  Enums:
    SourceTypeValueValuesEnum: Type of the model source.

  Fields:
    copy: If this Model is copy of another Model. If true then source_type
      pertains to the original.
    sourceType: Type of the model source.
  """

    class SourceTypeValueValuesEnum(_messages.Enum):
        """Type of the model source.

    Values:
      MODEL_SOURCE_TYPE_UNSPECIFIED: Should not be used.
      AUTOML: The Model is uploaded by automl training pipeline.
      CUSTOM: The Model is uploaded by user or custom training pipeline.
      BQML: The Model is registered and sync'ed from BigQuery ML.
      MODEL_GARDEN: The Model is saved or tuned from Model Garden.
    """
        MODEL_SOURCE_TYPE_UNSPECIFIED = 0
        AUTOML = 1
        CUSTOM = 2
        BQML = 3
        MODEL_GARDEN = 4
    copy = _messages.BooleanField(1)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 2)