from collections.abc import Mapping
import operator
import numpy as np
from .base import product
from .compat import filename_encode
from .. import h5z, h5p, h5d, h5f
 Check if chunks/maxshape match dataset rank 