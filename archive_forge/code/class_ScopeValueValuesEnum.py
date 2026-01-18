from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScopeValueValuesEnum(_messages.Enum):
    """The Scope metric captures whether a vulnerability in one vulnerable
    component impacts resources in components beyond its security scope.

    Values:
      SCOPE_UNSPECIFIED: Invalid value.
      SCOPE_UNCHANGED: An exploited vulnerability can only affect resources
        managed by the same security authority.
      SCOPE_CHANGED: An exploited vulnerability can affect resources beyond
        the security scope managed by the security authority of the vulnerable
        component.
    """
    SCOPE_UNSPECIFIED = 0
    SCOPE_UNCHANGED = 1
    SCOPE_CHANGED = 2