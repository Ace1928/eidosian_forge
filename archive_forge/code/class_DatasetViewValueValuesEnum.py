from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatasetViewValueValuesEnum(_messages.Enum):
    """Optional. Specifies the view that determines which dataset information
    is returned. By default, metadata and ACL information are returned.

    Values:
      DATASET_VIEW_UNSPECIFIED: The default value. Default to the FULL view.
      METADATA: Includes metadata information for the dataset, such as
        location, etag, lastModifiedTime, etc.
      ACL: Includes ACL information for the dataset, which defines dataset
        access for one or more entities.
      FULL: Includes both dataset metadata and ACL information.
    """
    DATASET_VIEW_UNSPECIFIED = 0
    METADATA = 1
    ACL = 2
    FULL = 3