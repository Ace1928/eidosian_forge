from scipy import fft
import numpy as np
import pytest
from numpy.testing import assert_allclose
import multiprocessing
import os
def _mt_fft(x):
    return fft.fft(x, workers=2)