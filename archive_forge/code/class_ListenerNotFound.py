import inspect
import sys
class ListenerNotFound(BrokerResponseError):
    errno = 72
    message = 'LISTENER_NOT_FOUND'
    description = 'There is no listener on the leader broker that matches the listener on which metadata request was processed'