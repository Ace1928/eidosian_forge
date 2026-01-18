import inspect
import sys
class InvalidFetchRequestError(BrokerResponseError):
    errno = 4
    message = 'INVALID_FETCH_SIZE'
    description = 'The message has a negative size.'