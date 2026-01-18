from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseTypeValueValuesEnum(_messages.Enum):
    """The type of the Cloud Firestore or Cloud Datastore database associated
    with this application.

    Values:
      DATABASE_TYPE_UNSPECIFIED: Database type is unspecified.
      CLOUD_DATASTORE: Cloud Datastore
      CLOUD_FIRESTORE: Cloud Firestore Native
      CLOUD_DATASTORE_COMPATIBILITY: Cloud Firestore in Datastore Mode
    """
    DATABASE_TYPE_UNSPECIFIED = 0
    CLOUD_DATASTORE = 1
    CLOUD_FIRESTORE = 2
    CLOUD_DATASTORE_COMPATIBILITY = 3