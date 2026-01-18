import inspect
import sys
class DelegationTokenAuthDisabled(BrokerResponseError):
    errno = 61
    message = 'DELEGATION_TOKEN_AUTH_DISABLED'
    description = 'Delegation Token feature is not enabled'