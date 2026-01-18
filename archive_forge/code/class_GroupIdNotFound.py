import inspect
import sys
class GroupIdNotFound(BrokerResponseError):
    errno = 69
    message = 'GROUP_ID_NOT_FOUND'
    description = 'The group id does not exist'