from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InclusionModeValueValuesEnum(_messages.Enum):
    """The mode to use for filtering asset discovery.

    Values:
      INCLUSION_MODE_UNSPECIFIED: Unspecified. Setting the mode with this
        value will disable inclusion/exclusion filtering for Asset Discovery.
      INCLUDE_ONLY: Asset Discovery will capture only the resources within the
        projects specified. All other resources will be ignored.
      EXCLUDE: Asset Discovery will ignore all resources under the projects
        specified. All other resources will be retrieved.
    """
    INCLUSION_MODE_UNSPECIFIED = 0
    INCLUDE_ONLY = 1
    EXCLUDE = 2