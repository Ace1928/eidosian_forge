import inspect
import sys
class InvalidRequiredAcksError(BrokerResponseError):
    errno = 21
    message = 'INVALID_REQUIRED_ACKS'
    description = 'Returned from a produce request if the requested requiredAcks is invalid (anything other than -1, 1, or 0).'