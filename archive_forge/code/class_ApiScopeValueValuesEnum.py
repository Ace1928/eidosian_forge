from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiScopeValueValuesEnum(_messages.Enum):
    """The API scope supported by this index.

    Values:
      ANY_API: The index can only be used by the Firestore Native query API.
        This is the default.
      DATASTORE_MODE_API: The index can only be used by the Firestore in
        Datastore Mode query API.
    """
    ANY_API = 0
    DATASTORE_MODE_API = 1