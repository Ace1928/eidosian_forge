import inspect
import sys
class ReplicaNotAvailableError(BrokerResponseError):
    errno = 9
    message = 'REPLICA_NOT_AVAILABLE'
    description = 'If replica is expected on a broker, but is not (this can be safely ignored).'