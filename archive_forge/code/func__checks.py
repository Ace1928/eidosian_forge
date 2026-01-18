import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError
def _checks(func):
    self.checkers[format] = (func, raises)
    return func