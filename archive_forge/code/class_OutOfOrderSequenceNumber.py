import inspect
import sys
class OutOfOrderSequenceNumber(BrokerResponseError):
    errno = 45
    message = 'OUT_OF_ORDER_SEQUENCE_NUMBER'
    description = 'The broker received an out of order sequence number'