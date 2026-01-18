from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppEngineIntegrationModeValueValuesEnum(_messages.Enum):
    """The App Engine integration mode to use for this database.

    Values:
      APP_ENGINE_INTEGRATION_MODE_UNSPECIFIED: Not used.
      ENABLED: If an App Engine application exists in the same region as this
        database, App Engine configuration will impact this database. This
        includes disabling of the application & database, as well as disabling
        writes to the database.
      DISABLED: App Engine has no effect on the ability of this database to
        serve requests. This is the default setting for databases created with
        the Firestore API.
    """
    APP_ENGINE_INTEGRATION_MODE_UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2