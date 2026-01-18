from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowStaticRoutesValueValuesEnum(_messages.Enum):
    """Specifies whether static route creation is allowed.

    Values:
      STATIC_ROUTES_ALLOWED: <no description>
      STATIC_ROUTES_BLOCKED: <no description>
      STATIC_ROUTES_UNSPECIFIED: <no description>
    """
    STATIC_ROUTES_ALLOWED = 0
    STATIC_ROUTES_BLOCKED = 1
    STATIC_ROUTES_UNSPECIFIED = 2