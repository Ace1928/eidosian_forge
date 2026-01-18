from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1TrainProcessorVersionRequestCustomDocumentExtractionOptions(_messages.Message):
    """Options to control the training of the Custom Document Extraction (CDE)
  Processor.

  Enums:
    TrainingMethodValueValuesEnum: Training method to use for CDE training.

  Fields:
    trainingMethod: Training method to use for CDE training.
  """

    class TrainingMethodValueValuesEnum(_messages.Enum):
        """Training method to use for CDE training.

    Values:
      TRAINING_METHOD_UNSPECIFIED: <no description>
      MODEL_BASED: <no description>
      TEMPLATE_BASED: <no description>
    """
        TRAINING_METHOD_UNSPECIFIED = 0
        MODEL_BASED = 1
        TEMPLATE_BASED = 2
    trainingMethod = _messages.EnumField('TrainingMethodValueValuesEnum', 1)