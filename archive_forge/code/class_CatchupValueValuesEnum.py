from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CatchupValueValuesEnum(_messages.Enum):
    """Whether the catchup is enabled for the DAG.

    Values:
      CATCHUP_VALUE_UNSPECIFIED: The state of the Cachup is unknown.
      ENABLED: The catchup is enabled.
      DISABLED: The catchup is disabled.
    """
    CATCHUP_VALUE_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2