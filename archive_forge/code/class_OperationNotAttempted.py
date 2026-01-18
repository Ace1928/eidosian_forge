import inspect
import sys
class OperationNotAttempted(BrokerResponseError):
    errno = 55
    message = 'OPERATION_NOT_ATTEMPTED'
    description = 'The broker did not attempt to execute this operation. This may happen for batched RPCs where some operations in the batch failed, causing the broker to respond without trying the rest.'