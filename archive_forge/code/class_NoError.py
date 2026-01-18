import inspect
import sys
class NoError(BrokerResponseError):
    errno = 0
    message = 'NO_ERROR'
    description = 'No error--it worked!'