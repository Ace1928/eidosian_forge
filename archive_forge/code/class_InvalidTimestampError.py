import inspect
import sys
class InvalidTimestampError(BrokerResponseError):
    errno = 32
    message = 'INVALID_TIMESTAMP'
    description = 'The timestamp of the message is out of acceptable range.'