import inspect
import sys
class DelegationTokenOwnerMismatch(BrokerResponseError):
    errno = 63
    message = 'DELEGATION_TOKEN_OWNER_MISMATCH'
    description = 'Specified Principal is not valid Owner/Renewer.'