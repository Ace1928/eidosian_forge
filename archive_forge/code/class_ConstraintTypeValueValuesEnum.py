from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConstraintTypeValueValuesEnum(_messages.Enum):
    """Describes the different types of constraints that are applied.

    Values:
      CONSTRAINT_TYPE_UNSPECIFIED: Unspecified constraint applied.
      RESOURCE_LOCATIONS_ORG_POLICY_CREATE_CONSTRAINT: The project's org
        policy disallows the given region.
    """
    CONSTRAINT_TYPE_UNSPECIFIED = 0
    RESOURCE_LOCATIONS_ORG_POLICY_CREATE_CONSTRAINT = 1