import inspect
import sys
class BrokerNotAvailableError(BrokerResponseError):
    errno = 8
    message = 'BROKER_NOT_AVAILABLE'
    description = 'This is not a client facing error and is used mostly by tools when a broker is not alive.'