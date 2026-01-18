import inspect
import sys
class IllegalSaslStateError(BrokerResponseError):
    errno = 34
    message = 'ILLEGAL_SASL_STATE'
    description = 'Request is not valid given the current SASL state.'