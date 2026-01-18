import inspect
import sys
class SecurityDisabled(BrokerResponseError):
    errno = 54
    message = 'SECURITY_DISABLED'
    description = 'Security features are disabled'