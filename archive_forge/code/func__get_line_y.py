import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_line_y(self, lineno):
    """
        Get the Y coordinate of a line number.
        """
    return lineno * self._get_line_height() + self.image_pad