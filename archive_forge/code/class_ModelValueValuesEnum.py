from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelValueValuesEnum(_messages.Enum):
    """The hardware form factor of the appliance.

    Values:
      TYPE_UNSPECIFIED: Default value. This value is unused.
      TA40_RACKABLE: A rackable TA40.
      TA40_STANDALONE: A standalone TA40.
      TA300_RACKABLE: A rackable TA300.
      TA300_STANDALONE: A standalone TA300.
      TA7: A TA7.
      EA_STORAGE_7: The storage-heavy Edge Appliance.
      EA_GPU_T4: The T4 GPU-capable Edge Appliance.
    """
    TYPE_UNSPECIFIED = 0
    TA40_RACKABLE = 1
    TA40_STANDALONE = 2
    TA300_RACKABLE = 3
    TA300_STANDALONE = 4
    TA7 = 5
    EA_STORAGE_7 = 6
    EA_GPU_T4 = 7