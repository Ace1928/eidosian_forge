import warnings
import numpy as np
from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
class Hdf5Bunch:
    """Make object for accessing attributes of variable"""

    def __init__(self, var):
        for name, value in var.attrs.items():
            setattr(self, name, value)