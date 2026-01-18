import inspect
import sys
class CorruptRecordException(BrokerResponseError):
    errno = 2
    message = 'CORRUPT_MESSAGE'
    description = 'This message has failed its CRC checksum, exceeds the valid size, or is otherwise corrupt.'