import inspect
import sys
class InvalidCommitOffsetSizeError(BrokerResponseError):
    errno = 28
    message = 'INVALID_COMMIT_OFFSET_SIZE'
    description = 'This error indicates that an offset commit was rejected because of oversize metadata.'