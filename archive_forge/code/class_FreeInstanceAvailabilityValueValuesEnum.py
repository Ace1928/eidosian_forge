from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FreeInstanceAvailabilityValueValuesEnum(_messages.Enum):
    """Output only. Describes whether free instances are available to be
    created in this instance config.

    Values:
      FREE_INSTANCE_AVAILABILITY_UNSPECIFIED: Not specified.
      AVAILABLE: Indicates that free instances are available to be created in
        this instance config.
      UNSUPPORTED: Indicates that free instances are not supported in this
        instance config.
      DISABLED: Indicates that free instances are currently not available to
        be created in this instance config.
      QUOTA_EXCEEDED: Indicates that additional free instances cannot be
        created in this instance config because the project has reached its
        limit of free instances.
    """
    FREE_INSTANCE_AVAILABILITY_UNSPECIFIED = 0
    AVAILABLE = 1
    UNSUPPORTED = 2
    DISABLED = 3
    QUOTA_EXCEEDED = 4