from typing import Union
from warnings import warn
from .low_level import *
class DBusErrorResponse(Exception):
    """Raised by proxy method calls when the reply is an error message"""

    def __init__(self, msg):
        self.name = msg.header.fields.get(HeaderFields.error_name)
        self.data = msg.body

    def __str__(self):
        return '[{}] {}'.format(self.name, self.data)