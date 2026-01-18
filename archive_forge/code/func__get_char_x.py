import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_char_x(self, charno):
    """
        Get the X coordinate of a character position.
        """
    return charno * self.fontw + self.image_pad + self.line_number_width