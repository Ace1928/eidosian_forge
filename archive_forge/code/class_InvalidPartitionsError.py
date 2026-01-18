import inspect
import sys
class InvalidPartitionsError(BrokerResponseError):
    errno = 37
    message = 'INVALID_PARTITIONS'
    description = 'Number of partitions is invalid.'