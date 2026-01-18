import inspect
import sys
class InvalidGroupIdError(BrokerResponseError):
    errno = 24
    message = 'INVALID_GROUP_ID'
    description = 'Returned in join group when the groupId is empty or null.'