import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_text_pos(self, charno, lineno):
    """
        Get the actual position for a character and line position.
        """
    return (self._get_char_x(charno), self._get_line_y(lineno))