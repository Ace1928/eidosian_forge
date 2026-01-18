import os
import functools
import operator
from scipy._lib import _pep440
import numpy as np
from numpy.testing import assert_
import pytest
import scipy.special as sc
class MissingModule:

    def __init__(self, name):
        self.name = name