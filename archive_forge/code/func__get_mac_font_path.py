import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _get_mac_font_path(self, font_map, name, style):
    return font_map.get((name + ' ' + style).strip().lower())