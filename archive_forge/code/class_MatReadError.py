import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
class MatReadError(Exception):
    """Exception indicating a read issue."""