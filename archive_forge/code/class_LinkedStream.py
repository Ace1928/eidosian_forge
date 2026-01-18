import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
class LinkedStream(Stream):
    """
    A LinkedStream indicates is automatically linked to plot interactions
    on a backend via a Renderer. Not all backends may support dynamically
    supplying stream data.
    """

    def __init__(self, linked=True, **params):
        super().__init__(linked=linked, **params)