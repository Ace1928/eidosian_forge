from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnvironmentSizeValueValuesEnum(_messages.Enum):
    """Optional. The size of the Cloud Composer environment. This field is
    supported for Cloud Composer environments in versions
    composer-2.*.*-airflow-*.*.* and newer.

    Values:
      ENVIRONMENT_SIZE_UNSPECIFIED: The size of the environment is
        unspecified.
      ENVIRONMENT_SIZE_SMALL: The environment size is small.
      ENVIRONMENT_SIZE_MEDIUM: The environment size is medium.
      ENVIRONMENT_SIZE_LARGE: The environment size is large.
    """
    ENVIRONMENT_SIZE_UNSPECIFIED = 0
    ENVIRONMENT_SIZE_SMALL = 1
    ENVIRONMENT_SIZE_MEDIUM = 2
    ENVIRONMENT_SIZE_LARGE = 3