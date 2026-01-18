import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_text_color(self, style):
    """
        Get the correct color for the token from the style.
        """
    if style['color'] is not None:
        fill = '#' + style['color']
    else:
        fill = '#000'
    return fill