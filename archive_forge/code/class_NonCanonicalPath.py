import logging
import re
import warnings
from inspect import getmembers, ismethod
from webob import exc
from .secure import handle_security, cross_boundary
from .util import iscontroller, getargspec, _cfg
class NonCanonicalPath(Exception):
    """
    Exception Raised when a non-canonical path is encountered when 'walking'
    the URI.  This is typically a ``POST`` request which requires a trailing
    slash.
    """

    def __init__(self, controller, remainder):
        self.controller = controller
        self.remainder = remainder