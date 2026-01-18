import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class ThrowingFloat(np.ndarray):

    def __float__(self):
        raise TypeError