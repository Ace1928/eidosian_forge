from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultStorageTypeValueValuesEnum(_messages.Enum):
    """The `StorageType` of the current instance. If unspecified, it will
    default to the first StorageType in the list of allowed_storage_types in
    the InstanceConfig for this instance.

    Values:
      STORAGE_TYPE_UNSPECIFIED: Storage type not specified.
      SSD: Flash (SSD) storage should be used.
      HDD: Magnetic drive (HDD) storage should be used.
    """
    STORAGE_TYPE_UNSPECIFIED = 0
    SSD = 1
    HDD = 2