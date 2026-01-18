import re
from . import errors
class InvalidPattern(errors.BzrError):
    _fmt = 'Invalid pattern(s) found. %(msg)s'

    def __init__(self, msg):
        self.msg = msg