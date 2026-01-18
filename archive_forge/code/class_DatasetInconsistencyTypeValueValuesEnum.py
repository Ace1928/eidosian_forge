from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetInconsistencyTypeValueValuesEnum(_messages.Enum):
    """The type of the inconsistency of the dataset.

    Values:
      DATASET_INCONSISTENCY_TYPE_UNSPECIFIED: Default value.
      DATASET_INCONSISTENCY_TYPE_NO_STORAGE_MARKER: The marker file under the
        dataset folder is not found.
    """
    DATASET_INCONSISTENCY_TYPE_UNSPECIFIED = 0
    DATASET_INCONSISTENCY_TYPE_NO_STORAGE_MARKER = 1