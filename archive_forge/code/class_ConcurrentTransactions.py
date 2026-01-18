import inspect
import sys
class ConcurrentTransactions(BrokerResponseError):
    errno = 51
    message = 'CONCURRENT_TRANSACTIONS'
    description = 'The producer attempted to update a transaction while another concurrent operation on the same transaction was ongoing'