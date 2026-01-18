import inspect
import sys
class GroupCoordinatorNotAvailableError(BrokerResponseError):
    errno = 15
    message = 'COORDINATOR_NOT_AVAILABLE'
    description = 'The broker returns this error code for group coordinator requests, offset commits, and most group management requests if the offsets topic has not yet been created, or if the group coordinator is not active.'
    retriable = True