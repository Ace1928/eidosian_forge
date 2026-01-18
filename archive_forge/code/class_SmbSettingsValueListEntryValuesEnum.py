from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SmbSettingsValueListEntryValuesEnum(_messages.Enum):
    """SmbSettingsValueListEntryValuesEnum enum type.

    Values:
      SMB_SETTINGS_UNSPECIFIED: Unspecified default option
      ENCRYPT_DATA: SMB setting encrypt data
      BROWSABLE: SMB setting browsable
      CHANGE_NOTIFY: SMB setting notify change
      NON_BROWSABLE: SMB setting not to notify change
      OPLOCKS: SMB setting oplocks
      SHOW_SNAPSHOT: SMB setting to show snapshots
      SHOW_PREVIOUS_VERSIONS: SMB setting to show previous versions
      ACCESS_BASED_ENUMERATION: SMB setting to access volume based on
        enumerartion
      CONTINUOUSLY_AVAILABLE: Continuously available enumeration
    """
    SMB_SETTINGS_UNSPECIFIED = 0
    ENCRYPT_DATA = 1
    BROWSABLE = 2
    CHANGE_NOTIFY = 3
    NON_BROWSABLE = 4
    OPLOCKS = 5
    SHOW_SNAPSHOT = 6
    SHOW_PREVIOUS_VERSIONS = 7
    ACCESS_BASED_ENUMERATION = 8
    CONTINUOUSLY_AVAILABLE = 9