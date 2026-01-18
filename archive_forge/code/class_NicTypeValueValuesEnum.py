from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NicTypeValueValuesEnum(_messages.Enum):
    """Optional. The type of vNIC to be used on this interface. This may be
    gVNIC or VirtioNet.

    Values:
      NIC_TYPE_UNSPECIFIED: Default should be NIC_TYPE_UNSPECIFIED.
      VIRTIO_NET: VIRTIO
      GVNIC: GVNIC
    """
    NIC_TYPE_UNSPECIFIED = 0
    VIRTIO_NET = 1
    GVNIC = 2