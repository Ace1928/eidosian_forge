from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
class item_check:
    """
    Context manager to allow creating NdMapping types without
    performing the usual item_checks, providing significant
    speedups when there are a lot of items. Should only be
    used when both keys and values are guaranteed to be the
    right type, as is the case for many internal operations.
    """

    def __init__(self, enabled):
        self.enabled = enabled

    def __enter__(self):
        self._enabled = MultiDimensionalMapping._check_items
        MultiDimensionalMapping._check_items = self.enabled

    def __exit__(self, exc_type, exc_val, exc_tb):
        MultiDimensionalMapping._check_items = self._enabled