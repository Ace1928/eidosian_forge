import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
class ErrorWithNoFormat(errors.BzrError):
    __doc__ = 'This class has a docstring but no format string.'