from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorCodeValueValuesEnum(_messages.Enum):
    """Output only. Code describing any errors that may have occurred.

    Values:
      ERROR_CODE_UNSPECIFIED: No error code was specified.
      CLOUD_BUILD_PERMISSION_DENIED: Cloud Build failed due to a permission
        issue.
      APPLY_BUILD_API_FAILED: Cloud Build job associated with creating or
        updating a deployment could not be started.
      APPLY_BUILD_RUN_FAILED: Cloud Build job associated with creating or
        updating a deployment was started but failed.
      QUOTA_VALIDATION_FAILED: quota validation failed for one or more
        resources in terraform configuration files.
    """
    ERROR_CODE_UNSPECIFIED = 0
    CLOUD_BUILD_PERMISSION_DENIED = 1
    APPLY_BUILD_API_FAILED = 2
    APPLY_BUILD_RUN_FAILED = 3
    QUOTA_VALIDATION_FAILED = 4