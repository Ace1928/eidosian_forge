import inspect
import sys
class RebalanceInProgressError(BrokerResponseError):
    errno = 27
    message = 'REBALANCE_IN_PROGRESS'
    description = 'Returned in heartbeat requests when the coordinator has begun rebalancing the group. This indicates to the client that it should rejoin the group.'