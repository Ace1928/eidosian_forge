from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IlbRouteBehaviorOnUnhealthyValueValuesEnum(_messages.Enum):
    """ILB route behavior when ILB is deemed unhealthy based on user
    specified threshold on the Backend Service of the internal load balancing.

    Values:
      DO_NOT_WITHDRAW_ROUTE_IF_ILB_UNHEALTHY: Do not Withdraw route if the ILB
        is deemed unhealthy based on user specified threshold on the Backend
        Service of the ILB. This is default behavior for ilb as next hop route
        without IlbRouteBehavior.
      WITHDRAW_ROUTE_IF_ILB_UNHEALTHY: Withdraw route if the ILB is deemed
        unhealthy based on user specified threshold on the Backend Service of
        the internal load balancing. Currently the withdrawn route will be
        reinserted when the backends are restored to healthy. If you wish to
        prevent the re-insertion of the route and trigger the fall-back at
        your discretion, override the health result from the backends to
        signal as healthy only when ready to fallback.
    """
    DO_NOT_WITHDRAW_ROUTE_IF_ILB_UNHEALTHY = 0
    WITHDRAW_ROUTE_IF_ILB_UNHEALTHY = 1