from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NonCriticalNodeErrorsValueListEntryValuesEnum(_messages.Enum):
    """NonCriticalNodeErrorsValueListEntryValuesEnum enum type.

    Values:
      NODE_ERROR_UNSPECIFIED: Reserved
      NODE_ERROR_INTERNAL_ERROR: Internal error If there is no matching error
        below, use it by default
      NODE_ERROR_DEVICE_NOT_FOUND: Device not found
      NODE_ERROR_DEVICE_STALE: Device is out of sync
      NODE_ERROR_DEVICE_CROSS_ORG: It is a cross-org device
      NODE_ERROR_DEVICE_INFO_NOT_AUTHORIZED: Caller doesn't have permission to
        device info
      NODE_ERROR_INVALID_SOURCE_IP: Source ip is not valid, from inIpRange
        function
      NODE_ERROR_INVALID_IP_SUBNETS: Ip subnets are not valid, from inIpRange
        function
      NODE_ERROR_INVALID_DEVICE_VERSION: Device min verion is not valid, from
        versionAtLeast function
      NODE_ERROR_NO_MATCHING_OVERLOADED_FUNC: Expr error from a supported
        function type with invalid parameters e.g. 1 == true
      NODE_ERROR_AUTH_SESSION_INFO_NOT_AUTHORIZED: Caller doesn't have
        permission to auth session info
      NODE_ERROR_NO_BCE_LICENSE: User is not assigned a BCE license.
    """
    NODE_ERROR_UNSPECIFIED = 0
    NODE_ERROR_INTERNAL_ERROR = 1
    NODE_ERROR_DEVICE_NOT_FOUND = 2
    NODE_ERROR_DEVICE_STALE = 3
    NODE_ERROR_DEVICE_CROSS_ORG = 4
    NODE_ERROR_DEVICE_INFO_NOT_AUTHORIZED = 5
    NODE_ERROR_INVALID_SOURCE_IP = 6
    NODE_ERROR_INVALID_IP_SUBNETS = 7
    NODE_ERROR_INVALID_DEVICE_VERSION = 8
    NODE_ERROR_NO_MATCHING_OVERLOADED_FUNC = 9
    NODE_ERROR_AUTH_SESSION_INFO_NOT_AUTHORIZED = 10
    NODE_ERROR_NO_BCE_LICENSE = 11