from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceTypeValueValuesEnum(_messages.Enum):
    """Required. The type of the Google Drive resource.

    Values:
      RESOURCE_TYPE_UNSPECIFIED: Unspecified resource type.
      RESOURCE_TYPE_FILE: File resource type.
      RESOURCE_TYPE_FOLDER: Folder resource type.
    """
    RESOURCE_TYPE_UNSPECIFIED = 0
    RESOURCE_TYPE_FILE = 1
    RESOURCE_TYPE_FOLDER = 2