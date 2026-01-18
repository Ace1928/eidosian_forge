from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProviderValueValuesEnum(_messages.Enum):
    """The database provider.

    Values:
      DATABASE_PROVIDER_UNSPECIFIED: The database provider is unknown.
      CLOUDSQL: CloudSQL runs the database.
      RDS: RDS runs the database.
    """
    DATABASE_PROVIDER_UNSPECIFIED = 0
    CLOUDSQL = 1
    RDS = 2