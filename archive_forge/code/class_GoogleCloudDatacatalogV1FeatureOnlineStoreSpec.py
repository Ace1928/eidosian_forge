from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1FeatureOnlineStoreSpec(_messages.Message):
    """Detail description of the source information of a Vertex Feature Online
  Store.

  Enums:
    StorageTypeValueValuesEnum: Output only. Type of underelaying storage for
      the FeatureOnlineStore.

  Fields:
    storageType: Output only. Type of underelaying storage for the
      FeatureOnlineStore.
  """

    class StorageTypeValueValuesEnum(_messages.Enum):
        """Output only. Type of underelaying storage for the FeatureOnlineStore.

    Values:
      STORAGE_TYPE_UNSPECIFIED: Should not be used.
      BIGTABLE: Underlsying storgae is Bigtable.
      OPTIMIZED: Underlaying is optimized online server (Lightning).
    """
        STORAGE_TYPE_UNSPECIFIED = 0
        BIGTABLE = 1
        OPTIMIZED = 2
    storageType = _messages.EnumField('StorageTypeValueValuesEnum', 1)