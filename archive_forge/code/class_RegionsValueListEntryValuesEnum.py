from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionsValueListEntryValuesEnum(_messages.Enum):
    """RegionsValueListEntryValuesEnum enum type.

    Values:
      REGION_UNSPECIFIED: no region
      us_central1: us-central1 region
      us_west1: us-west1 region
      us_east1: us-east1 region
      us_east4: us-east4 region
    """
    REGION_UNSPECIFIED = 0
    us_central1 = 1
    us_west1 = 2
    us_east1 = 3
    us_east4 = 4