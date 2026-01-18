from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PredefinedRoleValueValuesEnum(_messages.Enum):
    """predefined_role is the Kubernetes default role to use

    Values:
      UNKNOWN: UNKNOWN
      ADMIN: ADMIN has EDIT and RBAC permissions
      EDIT: EDIT can edit all resources except RBAC
      VIEW: VIEW can only read resources
      ANTHOS_SUPPORT: ANTHOS_SUPPORT gives Google Support read-only access to
        a number of cluster resources.
    """
    UNKNOWN = 0
    ADMIN = 1
    EDIT = 2
    VIEW = 3
    ANTHOS_SUPPORT = 4