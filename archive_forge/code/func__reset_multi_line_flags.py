import re
from mako import exceptions
def _reset_multi_line_flags(self):
    """reset the flags which would indicate we are in a backslashed
        or triple-quoted section."""
    self.backslashed, self.triplequoted = (False, False)