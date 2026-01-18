from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def _move_up(rot_amount, trans_amount):
    RF = trans_amount.parent()
    return unit_3_vector_and_distance_to_O13_hyperbolic_translation([RF(0), RF(+1), RF(0)], trans_amount)