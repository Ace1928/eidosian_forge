import inspect
import sys
class IllegalGenerationError(BrokerResponseError):
    errno = 22
    message = 'ILLEGAL_GENERATION'
    description = 'Returned from group membership requests (such as heartbeats) when the generation id provided in the request is not the current generation.'