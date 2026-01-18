from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmCapabilities(_messages.Message):
    """Migrating VM source information about the VM capabilities needed for
  some Compute Engine features.

  Enums:
    OsCapabilitiesValueListEntryValuesEnum:

  Fields:
    lastOsCapabilitiesUpdateTime: Output only. The last time OS capabilities
      list was updated.
    osCapabilities: Output only. Unordered list. List of certain VM OS
      capabilities needed for some Compute Engine features.
  """

    class OsCapabilitiesValueListEntryValuesEnum(_messages.Enum):
        """OsCapabilitiesValueListEntryValuesEnum enum type.

    Values:
      OS_CAPABILITY_UNSPECIFIED: This is for API compatibility only and is not
        in use.
      OS_CAPABILITY_NVME_STORAGE_ACCESS: NVMe driver installed and the VM can
        use NVMe PD or local SSD.
      OS_CAPABILITY_GVNIC_NETWORK_INTERFACE: gVNIC virtual NIC driver
        supported.
    """
        OS_CAPABILITY_UNSPECIFIED = 0
        OS_CAPABILITY_NVME_STORAGE_ACCESS = 1
        OS_CAPABILITY_GVNIC_NETWORK_INTERFACE = 2
    lastOsCapabilitiesUpdateTime = _messages.StringField(1)
    osCapabilities = _messages.EnumField('OsCapabilitiesValueListEntryValuesEnum', 2, repeated=True)