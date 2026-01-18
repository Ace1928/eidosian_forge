import inspect
import sys
class InconsistentGroupProtocolError(BrokerResponseError):
    errno = 23
    message = 'INCONSISTENT_GROUP_PROTOCOL'
    description = 'Returned in join group when the member provides a protocol type or set of protocols which is not compatible with the current group.'