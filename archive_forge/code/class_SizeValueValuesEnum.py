from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SizeValueValuesEnum(_messages.Enum):
    """Indicates the size of the backing VM running the environment. If set
    to something other than DEFAULT, it will be reverted to the default VM
    size after vm_size_expire_time.

    Values:
      VM_SIZE_UNSPECIFIED: The VM size is unknown.
      DEFAULT: The default VM size.
      BOOSTED: The boosted VM size.
    """
    VM_SIZE_UNSPECIFIED = 0
    DEFAULT = 1
    BOOSTED = 2