from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def _turn_ccw(rot_amount, trans_amount):
    return O13_z_rotation(rot_amount)