from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3BatchMoveDocumentsMetadata(_messages.Message):
    """A GoogleCloudDocumentaiUiv1beta3BatchMoveDocumentsMetadata object.

  Enums:
    DestDatasetTypeValueValuesEnum: The destination dataset split type.
    DestSplitTypeValueValuesEnum: The destination dataset split type.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    destDatasetType: The destination dataset split type.
    destSplitType: The destination dataset split type.
    individualBatchMoveStatuses: The list of response details of each
      document.
  """

    class DestDatasetTypeValueValuesEnum(_messages.Enum):
        """The destination dataset split type.

    Values:
      DATASET_SPLIT_TYPE_UNSPECIFIED: Default value if the enum is not set.
      DATASET_SPLIT_TRAIN: Identifies the train documents.
      DATASET_SPLIT_TEST: Identifies the test documents.
      DATASET_SPLIT_UNASSIGNED: Identifies the unassigned documents.
    """
        DATASET_SPLIT_TYPE_UNSPECIFIED = 0
        DATASET_SPLIT_TRAIN = 1
        DATASET_SPLIT_TEST = 2
        DATASET_SPLIT_UNASSIGNED = 3

    class DestSplitTypeValueValuesEnum(_messages.Enum):
        """The destination dataset split type.

    Values:
      DATASET_SPLIT_TYPE_UNSPECIFIED: Default value if the enum is not set.
      DATASET_SPLIT_TRAIN: Identifies the train documents.
      DATASET_SPLIT_TEST: Identifies the test documents.
      DATASET_SPLIT_UNASSIGNED: Identifies the unassigned documents.
    """
        DATASET_SPLIT_TYPE_UNSPECIFIED = 0
        DATASET_SPLIT_TRAIN = 1
        DATASET_SPLIT_TEST = 2
        DATASET_SPLIT_UNASSIGNED = 3
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)
    destDatasetType = _messages.EnumField('DestDatasetTypeValueValuesEnum', 2)
    destSplitType = _messages.EnumField('DestSplitTypeValueValuesEnum', 3)
    individualBatchMoveStatuses = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3BatchMoveDocumentsMetadataIndividualBatchMoveStatus', 4, repeated=True)