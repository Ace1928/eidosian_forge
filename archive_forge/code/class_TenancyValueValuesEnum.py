from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TenancyValueValuesEnum(_messages.Enum):
    """Required. The tenancy for instance.

    Values:
      TENANCY_UNSPECIFIED: Not set.
      DEFAULT: Use default VPC tenancy.
      DEDICATED: Run a dedicated instance.
      HOST: Launch this instance to a dedicated host.
    """
    TENANCY_UNSPECIFIED = 0
    DEFAULT = 1
    DEDICATED = 2
    HOST = 3