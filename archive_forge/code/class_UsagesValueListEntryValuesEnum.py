from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsagesValueListEntryValuesEnum(_messages.Enum):
    """UsagesValueListEntryValuesEnum enum type.

    Values:
      EXECUTION_ENVIRONMENT_USAGE_UNSPECIFIED: Default value. This value is
        unused.
      RENDER: Use for rendering.
      DEPLOY: Use for deploying and deployment hooks.
      VERIFY: Use for deployment verification.
      PREDEPLOY: Use for predeploy job execution.
      POSTDEPLOY: Use for postdeploy job execution.
    """
    EXECUTION_ENVIRONMENT_USAGE_UNSPECIFIED = 0
    RENDER = 1
    DEPLOY = 2
    VERIFY = 3
    PREDEPLOY = 4
    POSTDEPLOY = 5