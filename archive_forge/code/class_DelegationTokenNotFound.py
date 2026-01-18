import inspect
import sys
class DelegationTokenNotFound(BrokerResponseError):
    errno = 62
    message = 'DELEGATION_TOKEN_NOT_FOUND'
    description = 'Delegation Token is not found on server.'