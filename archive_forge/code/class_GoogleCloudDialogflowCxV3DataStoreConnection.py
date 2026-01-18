from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3DataStoreConnection(_messages.Message):
    """A data store connection. It represents a data store in Discovery Engine
  and the type of the contents it contains.

  Enums:
    DataStoreTypeValueValuesEnum: The type of the connected data store.

  Fields:
    dataStore: The full name of the referenced data store. Formats: `projects/
      {project}/locations/{location}/collections/{collection}/dataStores/{data
      _store}`
      `projects/{project}/locations/{location}/dataStores/{data_store}`
    dataStoreType: The type of the connected data store.
  """

    class DataStoreTypeValueValuesEnum(_messages.Enum):
        """The type of the connected data store.

    Values:
      DATA_STORE_TYPE_UNSPECIFIED: Not specified. This value indicates that
        the data store type is not specified, so it will not be used during
        search.
      PUBLIC_WEB: A data store that contains public web content.
      UNSTRUCTURED: A data store that contains unstructured private data.
      STRUCTURED: A data store that contains structured data (for example
        FAQ).
    """
        DATA_STORE_TYPE_UNSPECIFIED = 0
        PUBLIC_WEB = 1
        UNSTRUCTURED = 2
        STRUCTURED = 3
    dataStore = _messages.StringField(1)
    dataStoreType = _messages.EnumField('DataStoreTypeValueValuesEnum', 2)