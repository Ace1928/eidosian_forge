import inspect
import sys
class TransactionCoordinatorFenced(BrokerResponseError):
    errno = 52
    message = 'TRANSACTION_COORDINATOR_FENCED'
    description = 'Indicates that the transaction coordinator sending a WriteTxnMarker is no longer the current coordinator for a given producer'