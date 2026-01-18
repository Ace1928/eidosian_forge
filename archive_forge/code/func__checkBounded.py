import numpy as np
import os
import time
from ..utils import *
def _checkBounded(xval, yval, w, h, mbSize):
    if yval < 0 or yval + mbSize >= h or xval < 0 or (xval + mbSize >= w):
        return False
    else:
        return True