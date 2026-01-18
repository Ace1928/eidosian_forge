from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteInstallStatusValueValuesEnum(_messages.Enum):
    """An enum representing the status of the route installation.

    Values:
      ROUTE_INSTALL_STATUS_UNSPECIFIED: The install status was not specified.
      ROUTE_INSTALL_STATUS_ACTIVE: The route was installed correctly and is
        active.
      ROUTE_INSTALL_STATUS_PENDING: The route has not been installed, but
        there's no error (for example, the route is in the process of being
        installed)
      ROUTE_INSTALL_STATUS_FAILED: The route installation failed due to an
        error.
    """
    ROUTE_INSTALL_STATUS_UNSPECIFIED = 0
    ROUTE_INSTALL_STATUS_ACTIVE = 1
    ROUTE_INSTALL_STATUS_PENDING = 2
    ROUTE_INSTALL_STATUS_FAILED = 3