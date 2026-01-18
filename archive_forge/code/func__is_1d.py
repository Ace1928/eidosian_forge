from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _is_1d(data):
    try:
        return len(data.shape) == 1
    except AttributeError:
        return True