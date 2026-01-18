from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def _turn_left(rot_amount, trans_amount):
    return O13_y_rotation(-rot_amount)