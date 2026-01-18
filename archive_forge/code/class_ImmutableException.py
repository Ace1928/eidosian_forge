import logging
import uuid
class ImmutableException(Exception):

    def __init__(self, attribute=None):
        message = 'This object is immutable!'
        super(ImmutableException, self).__init__(message)