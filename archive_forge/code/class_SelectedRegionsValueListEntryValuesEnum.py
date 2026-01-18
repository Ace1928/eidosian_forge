from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SelectedRegionsValueListEntryValuesEnum(_messages.Enum):
    """SelectedRegionsValueListEntryValuesEnum enum type.

    Values:
      REGION_UNSPECIFIED: Default value if no region is specified. Will result
        in Uptime checks running from all regions.
      USA: Allows checks to run from locations within the United States of
        America.
      EUROPE: Allows checks to run from locations within the continent of
        Europe.
      SOUTH_AMERICA: Allows checks to run from locations within the continent
        of South America.
      ASIA_PACIFIC: Allows checks to run from locations within the Asia
        Pacific area (ex: Singapore).
      USA_OREGON: Allows checks to run from locations within the western
        United States of America
      USA_IOWA: Allows checks to run from locations within the central United
        States of America
      USA_VIRGINIA: Allows checks to run from locations within the eastern
        United States of America
    """
    REGION_UNSPECIFIED = 0
    USA = 1
    EUROPE = 2
    SOUTH_AMERICA = 3
    ASIA_PACIFIC = 4
    USA_OREGON = 5
    USA_IOWA = 6
    USA_VIRGINIA = 7