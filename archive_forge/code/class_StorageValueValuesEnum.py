from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageValueValuesEnum(_messages.Enum):
    """Optional. Where provenance is stored.

    Values:
      STORAGE_UNSPECIFIED: Default PREFER_ARTIFACT_PROJECT.
      PREFER_ARTIFACT_PROJECT: GCB will attempt to push provenance to the
        artifact project. If it is not available, fallback to build project.
      ARTIFACT_PROJECT_ONLY: Only push to artifact project.
      BUILD_PROJECT_ONLY: Only push to build project.
    """
    STORAGE_UNSPECIFIED = 0
    PREFER_ARTIFACT_PROJECT = 1
    ARTIFACT_PROJECT_ONLY = 2
    BUILD_PROJECT_ONLY = 3