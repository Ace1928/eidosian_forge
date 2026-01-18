import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
class SpacegroupNotFoundError(SpacegroupError):
    """Raised when given space group cannot be found in data base."""
    pass