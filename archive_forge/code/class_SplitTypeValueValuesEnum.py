from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SplitTypeValueValuesEnum(_messages.Enum):
    """The dataset split type.

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