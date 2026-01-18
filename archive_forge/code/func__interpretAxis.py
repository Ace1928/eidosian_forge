import copy
import os
import pickle
import warnings
import numpy as np
def _interpretAxis(self, axis):
    if isinstance(axis, (str, tuple)):
        return self._getAxis(axis)
    else:
        return axis