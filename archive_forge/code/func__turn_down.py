from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def _turn_down(rot_amount, trans_amount):
    return O13_x_rotation(rot_amount)