from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RealmValueValuesEnum(_messages.Enum):
    """A realm in which the instance is deployed.

    Values:
      REALM_UNSPECIFIED: realm not specified
      REALM_NA_CENTRAL: us-central1
      REALM_NA_EAST: us-east[1|4]
      REALM_NA_WEST: us-west[1|2|4]
      REALM_ASIA_NORTHEAST: asia-northeast[1|3]
      REALM_ASIA_SOUTHEAST: asia-southeast[1|2]
      REALM_EU_WEST: europe-west[1-4]
    """
    REALM_UNSPECIFIED = 0
    REALM_NA_CENTRAL = 1
    REALM_NA_EAST = 2
    REALM_NA_WEST = 3
    REALM_ASIA_NORTHEAST = 4
    REALM_ASIA_SOUTHEAST = 5
    REALM_EU_WEST = 6