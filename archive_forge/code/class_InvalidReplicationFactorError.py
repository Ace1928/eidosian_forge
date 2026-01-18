import inspect
import sys
class InvalidReplicationFactorError(BrokerResponseError):
    errno = 38
    message = 'INVALID_REPLICATION_FACTOR'
    description = 'Replication-factor is invalid.'