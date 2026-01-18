import json
import re
import typing as ty
from requests import exceptions as _rex
class MethodNotSupported(SDKException):
    """The resource does not support this operation type."""

    def __init__(self, resource, method):
        try:
            name = resource.__name__
        except AttributeError:
            name = resource.__class__.__name__
        message = 'The %s method is not supported for %s.%s' % (method, resource.__module__, name)
        super(MethodNotSupported, self).__init__(message=message)