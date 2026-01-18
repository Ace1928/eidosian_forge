from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExportToSecurityCommandCenterValueValuesEnum(_messages.Enum):
    """Controls export of scan configurations and results to Security Command
    Center.

    Values:
      EXPORT_TO_SECURITY_COMMAND_CENTER_UNSPECIFIED: Use default, which is
        ENABLED.
      ENABLED: Export results of this scan to Security Command Center.
      DISABLED: Do not export results of this scan to Security Command Center.
    """
    EXPORT_TO_SECURITY_COMMAND_CENTER_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2