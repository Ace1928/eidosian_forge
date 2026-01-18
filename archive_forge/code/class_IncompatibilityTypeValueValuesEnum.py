from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IncompatibilityTypeValueValuesEnum(_messages.Enum):
    """The incompatibility type of this issue.

    Values:
      UNSPECIFIED: Default value, should not be used.
      INCOMPATIBILITY: Indicates that the issue is a known incompatibility
        between the cluster and Autopilot mode.
      ADDITIONAL_CONFIG_REQUIRED: Indicates the issue is an incompatibility if
        customers take no further action to resolve.
      PASSED_WITH_OPTIONAL_CONFIG: Indicates the issue is not an
        incompatibility, but depending on the workloads business logic, there
        is a potential that they won't work on Autopilot.
    """
    UNSPECIFIED = 0
    INCOMPATIBILITY = 1
    ADDITIONAL_CONFIG_REQUIRED = 2
    PASSED_WITH_OPTIONAL_CONFIG = 3