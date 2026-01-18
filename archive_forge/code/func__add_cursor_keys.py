from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def _add_cursor_keys(d):
    d['left'] = _turn_left
    d['right'] = _turn_right
    d['up'] = _turn_up
    d['down'] = _turn_down
    return d