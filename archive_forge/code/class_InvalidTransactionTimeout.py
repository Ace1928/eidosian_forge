import inspect
import sys
class InvalidTransactionTimeout(BrokerResponseError):
    errno = 50
    message = 'INVALID_TRANSACTION_TIMEOUT'
    description = 'The transaction timeout is larger than the maximum value allowed by the broker (as configured by transaction.max.timeout.ms).'