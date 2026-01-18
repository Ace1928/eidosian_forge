import inspect
import sys
class RequestTimedOutError(BrokerResponseError):
    errno = 7
    message = 'REQUEST_TIMED_OUT'
    description = 'This error is thrown if the request exceeds the user-specified time limit in the request.'
    retriable = True