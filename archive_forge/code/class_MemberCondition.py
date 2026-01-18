from libcloud.common.types import LibcloudError
class MemberCondition:
    """
    Each member of a load balancer can have an associated condition
    which determines its role within the load balancer.
    """
    ENABLED = 0
    DISABLED = 1
    DRAINING = 2