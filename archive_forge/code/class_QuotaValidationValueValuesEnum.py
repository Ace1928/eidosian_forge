from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaValidationValueValuesEnum(_messages.Enum):
    """Optional. Input to control quota checks for resources in terraform
    configuration files. There are limited resources on which quota validation
    applies.

    Values:
      QUOTA_VALIDATION_UNSPECIFIED: The default value. QuotaValidation on
        terraform configuration files will be disabled in this case.
      ENABLED: Enable computing quotas for resources in terraform
        configuration files to get visibility on resources with insufficient
        quotas.
      ENFORCED: Enforce quota checks so deployment fails if there isn't
        sufficient quotas available to deploy resources in terraform
        configuration files.
    """
    QUOTA_VALIDATION_UNSPECIFIED = 0
    ENABLED = 1
    ENFORCED = 2