import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
class SpacegroupError(Exception):
    """Base exception for the spacegroup module."""
    pass