from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterfaceTypesValueListEntryValuesEnum(_messages.Enum):
    """InterfaceTypesValueListEntryValuesEnum enum type.

    Values:
      GVNIC: GVNIC
      IDPF: IDPF
      RDMA: DEPRECATED: Please use TNA_IRDMA instead.
      UNSPECIFIED_NIC_TYPE: No type specified.
      VIRTIO_NET: VIRTIO
    """
    GVNIC = 0
    IDPF = 1
    RDMA = 2
    UNSPECIFIED_NIC_TYPE = 3
    VIRTIO_NET = 4