import inspect
import sys
class UnsupportedForMessageFormatError(BrokerResponseError):
    errno = 43
    message = 'UNSUPPORTED_FOR_MESSAGE_FORMAT'
    description = 'The message format version on the broker does not support this request.'