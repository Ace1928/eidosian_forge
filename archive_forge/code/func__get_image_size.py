import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_image_size(self, maxcharno, maxlineno):
    """
        Get the required image size.
        """
    return (self._get_char_x(maxcharno) + self.image_pad, self._get_line_y(maxlineno + 0) + self.image_pad)