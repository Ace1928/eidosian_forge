from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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